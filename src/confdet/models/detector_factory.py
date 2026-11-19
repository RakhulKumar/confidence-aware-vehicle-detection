"""Detectors: training, loading, raw-candidate prediction.

The project's contribution is the post-hoc layer, so this module is thin. It
does three things and guards against three failures seen in the first run:

  * train()            fine-tunes an Ultralytics model and copies the best
                       checkpoint to ONE canonical path. (Ultralytics renames
                       a run directory that already exists to "<name>2", so a
                       second training run used to leave every later stage
                       reading the first run's stale weights.)
  * VRAM guard         the first run reported 20.5 GB in use on a 16 GB card:
                       the batch did not fit, Windows spilled it into shared
                       system memory, and each iteration took 6.6 s instead
                       of well under one. Training now stops with an
                       explanation when reserved memory exceeds the card.
  * load_detector()    returns an object with .names and
                       .predict(image) -> (xyxy, confidence, class), always at
                       the near-zero confidence floor, image size, NMS IoU and
                       detection cap from the config, so every stage sees the
                       same raw candidates.

family: "yolo" (YOLO26 / YOLOv8 ...), "rtdetr", or "simulated" (the test
detector from evaluation/simulate.py, used by scripts/00_selftest.py).
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class VRAMOverflowError(RuntimeError):
    pass


@dataclass
class DetectorConfig:
    name: str
    family: str = "yolo"
    weights: str = "yolo26m.pt"
    imgsz: int = 960
    epochs: int = 100
    patience: int = 25
    batch: int = 8
    workers: int = 8
    optimizer: str = "auto"
    lr0: float | None = None
    train_args: dict = field(default_factory=dict)     # passed straight to Ultralytics .train()
    simulation: dict = field(default_factory=dict)     # SimConfig overrides (family: simulated)


def detector_config(cfg: dict, name: str | None = None) -> DetectorConfig:
    name = name or cfg["detectors"]["active"]
    models = cfg["detectors"]["models"]
    if name not in models:
        raise KeyError(f"detector {name!r} is not defined under detectors.models (have: {sorted(models)})")
    spec = dict(models[name])
    family = str(spec.pop("family", "yolo")).lower()
    if family == "ultralytics":                         # old config value
        family = "rtdetr" if "rtdetr" in str(spec.get("weights", "")).lower() else "yolo"
    known = {f.name for f in dataclasses.fields(DetectorConfig)} - {"name", "family"}
    unknown = set(spec) - known
    if unknown:
        raise ValueError(f"detector {name!r}: unknown config keys {sorted(unknown)}; "
                         f"put extra Ultralytics training arguments under train_args")
    return DetectorConfig(name=name, family=family, **spec)


def _ultralytics_class(family: str):
    try:
        import ultralytics
    except ImportError as e:  # pragma: no cover
        raise ImportError("this step needs the `ultralytics` package (pip install -r requirements.txt)") from e
    if family == "rtdetr":
        return ultralytics.RTDETR
    if family in ("yolo", "yolo_evidential"):            # yolo_evidential: models/evidential_yolo.py (v0.6 pilot)
        return ultralytics.YOLO
    raise ValueError(f"unknown detector family {family!r} (use yolo, yolo_evidential, rtdetr or simulated)")


# ---------------------------------------------------------------------------
# Sharing the GPU with another project (v0.6): a VRAM cap and a compute duty cycle
# ---------------------------------------------------------------------------
def apply_memory_fraction(fraction: float | None) -> bool:
    """Cap this process's CUDA caching allocator at `fraction` of the card
    (torch.cuda.set_per_process_memory_fraction). A batch that needs more
    fails with an out-of-memory error instead of taking the other project's
    memory. Returns whether a cap was set."""
    if not fraction:
        return False
    import torch
    if not torch.cuda.is_available():
        return False
    torch.cuda.set_per_process_memory_fraction(float(fraction), torch.cuda.current_device())
    return True


class DutyCycle:
    """Keeps this process's GPU busy at most `duty` of the time: after each unit
    of work (a training or validation batch, one prediction) it sleeps for
    elapsed * (1/duty - 1) -- i.e. as long as the work took, at duty = 0.5.
    `stats` records busy and slept seconds so the share can be checked."""

    def __init__(self, duty: float = 0.5, sleeper=time.sleep, clock=time.perf_counter):
        if not 0 < duty <= 1:
            raise ValueError(f"duty cycle must be in (0, 1], got {duty}")
        self.ratio = 1.0 / duty - 1.0
        self.sleep, self.clock = sleeper, clock
        self._t0 = None
        self.stats = {"busy_s": 0.0, "slept_s": 0.0, "units": 0}

    def start(self, *_):
        self._t0 = self.clock()

    def end(self, *_):
        if self._t0 is None:
            return
        busy = self.clock() - self._t0
        self._t0 = None
        pause = busy * self.ratio
        self.stats["busy_s"] += busy
        self.stats["slept_s"] += pause
        self.stats["units"] += 1
        if pause > 0:
            self.sleep(pause)

    def callbacks(self) -> dict:
        """Ultralytics events this throttle hooks (training and per-epoch validation)."""
        return {"on_train_batch_start": self.start, "on_train_batch_end": self.end,
                "on_val_batch_start": self.start, "on_val_batch_end": self.end}


def _vram_guard(abort: bool, logger, batch: int, limit_fraction: float | None = None):
    """Ultralytics callback: stop (or warn) when PyTorch has reserved more
    memory than the GPU physically has (or than this process's share of it,
    `limit_fraction`), i.e. the batch is being paged into shared system memory
    or is taking another process's memory.
    """
    state = {"n": 0, "warned": False}

    def on_train_batch_end(trainer):  # noqa: ARG001
        state["n"] += 1
        n = state["n"]
        if n < 10 or (n > 300 and n % 500):
            return
        try:
            import torch
            if not torch.cuda.is_available():
                return
            dev = torch.cuda.current_device()
            reserved = torch.cuda.memory_reserved(dev)
            total = torch.cuda.get_device_properties(dev).total_memory
        except Exception:
            return
        limit = total * (float(limit_fraction) if limit_fraction else 1.0)
        if reserved <= limit:
            return
        msg = (f"GPU memory overflow: PyTorch has reserved {reserved / 2**30:.1f} GB, above this "
               f"process's limit of {limit / 2**30:.1f} GB on a {total / 2**30:.1f} GB card. The batch "
               f"does not fit its share and is being paged or taken from other processes. Lower `batch` for this "
               f"detector in configs/experiment.yaml (currently {batch}; try {max(1, batch // 2)}) "
               f"and run again.")
        if abort:
            raise VRAMOverflowError(msg)
        if not state["warned"]:
            state["warned"] = True
            if logger:
                logger.warning(msg)

    return on_train_batch_end


def _locate_best(model, run_dir: Path) -> Path | None:
    candidates: list[Path] = []
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        if getattr(trainer, "best", None):
            candidates.append(Path(trainer.best))
        if getattr(trainer, "save_dir", None):
            candidates += [Path(trainer.save_dir) / "weights" / "best.pt",
                           Path(trainer.save_dir) / "weights" / "last.pt"]
    candidates += [run_dir / "train" / "weights" / "best.pt", run_dir / "train" / "weights" / "last.pt"]
    return next((p for p in candidates if p.exists()), None)


def last_checkpoint(run_dir: str | Path) -> Path:
    """Where Ultralytics writes the every-epoch checkpoint for our run layout."""
    return Path(run_dir).resolve() / "train" / "weights" / "last.pt"


def _checkpoint_finished(path: Path) -> bool:
    """Ultralytics marks a checkpoint of a COMPLETED run with epoch == -1."""
    try:
        import torch
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        return int(ck.get("epoch", 0)) == -1
    except Exception:
        return False


def train(det: DetectorConfig, data_yaml: str | Path, run_dir: str | Path, device: str = "cuda",
          seed: int = 42, abort_on_vram_overflow: bool = True, logger=None,
          resume: bool = False, gpu_share: dict | None = None) -> Path:
    """Fine-tune `det` on the materialised dataset and return the canonical
    checkpoint path  <run_dir>/weights/best.pt.

    resume: if <run_dir>/train/weights/last.pt exists, continue that run with
    Ultralytics resume=True (same arguments, seed and epoch counter, restored
    from the checkpoint) instead of starting again; a run that already finished
    is not retrained. Ultralytics writes last.pt after every epoch.
    gpu_share: {"memory_fraction": f, "duty_cycle": d} -- cap this process's
    VRAM at f of the card and keep the GPU busy at most d of the time.
    """
    run_dir = Path(run_dir).resolve()
    out = run_dir / "weights" / "best.pt"
    out.parent.mkdir(parents=True, exist_ok=True)

    if det.family == "simulated":
        # train_args.seed overrides the project seed here too, as it does for Ultralytics below
        sim_seed = int((det.train_args or {}).get("seed", seed))
        out.write_text(json.dumps({"family": "simulated", "simulation": det.simulation, "seed": sim_seed}))
        return out

    share = gpu_share or {}
    fraction = share.get("memory_fraction")
    apply_memory_fraction(fraction)
    duty = DutyCycle(float(share["duty_cycle"])) if share.get("duty_cycle") else None

    def _hook(model):
        model.add_callback("on_train_batch_end", _vram_guard(abort_on_vram_overflow, logger, det.batch, fraction))
        if duty is not None:
            for event, fn in duty.callbacks().items():
                model.add_callback(event, fn)
        if det.family == "yolo_evidential":
            from confdet.models import evidential_yolo
            model.add_callback("on_train_epoch_start", evidential_yolo.epoch_callback)   # KL annealing

    extra = {}
    if det.family == "yolo_evidential":
        from confdet.models import evidential_yolo
        extra["trainer"] = evidential_yolo.evidential_trainer()

    last = last_checkpoint(run_dir)
    if resume and last.exists():
        if _checkpoint_finished(last):
            if logger:
                logger.info(f"{last} belongs to a finished run; not retraining")
            best = last.parent / "best.pt"
            _atomic_copy(best if best.exists() else last, out)
            return out
        if logger:
            logger.info(f"resuming from {last} (Ultralytics resume=True: same arguments, seed and epoch counter)")
        model = _ultralytics_class(det.family)(str(last))
        _hook(model)
        model.train(resume=True, **extra)
    else:
        model = _ultralytics_class(det.family)(det.weights)
        _hook(model)
        kwargs = dict(
            data=str(Path(data_yaml).resolve()),
            imgsz=det.imgsz, epochs=det.epochs, patience=det.patience, batch=det.batch, workers=det.workers,
            # absolute `project`: Ultralytics re-roots a relative one under runs/detect/.
            # exist_ok: reuse the same folder instead of creating train2, train3, ...
            project=str(run_dir), name="train", exist_ok=True,
            device=device, seed=seed,
            save_period=-1,      # one rolling last.pt + best.pt per run; never per-epoch copies
        )
        if det.optimizer and det.optimizer != "auto":
            kwargs["optimizer"] = det.optimizer            # with "auto" Ultralytics ignores lr0
            if det.lr0 is not None:
                kwargs["lr0"] = det.lr0
        kwargs.update(det.train_args or {})
        model.train(**kwargs, **extra)
    if duty is not None and logger:
        s = duty.stats
        logger.info(f"duty cycle: {s['units']} GPU work units, busy {s['busy_s'] / 3600:.2f} h, "
                    f"paused {s['slept_s'] / 3600:.2f} h")

    best = _locate_best(model, run_dir)
    if best is None:
        raise FileNotFoundError(f"training finished but no checkpoint was found under {run_dir}")
    if best.name != "best.pt" and logger:
        logger.warning(f"best.pt not found; using {best}")
    _atomic_copy(best, out)
    return out


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy to a temporary name, then rename: a reader never sees half a file."""
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
_EMPTY = (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32))


class UltralyticsDetector:
    def __init__(self, weights: str | Path, family: str, imgsz: int, conf_floor: float,
                 nms_iou: float, max_det: int, duty_cycle: float | None = None):
        self.model = _ultralytics_class(family)(str(weights))
        self.duty = DutyCycle(float(duty_cycle)) if duty_cycle else None
        names = self.model.names
        self.names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        self.imgsz, self.conf_floor, self.nms_iou, self.max_det = imgsz, conf_floor, nms_iou, max_det

    def predict(self, source):
        """source: image path, or an HxWx3 BGR uint8 array. Returns every raw
        candidate at the confidence floor as (xyxy, confidence, class).
        """
        if isinstance(source, (str, Path)):
            source = str(source)
        if self.duty:
            self.duty.start()
        result = self.model.predict(source=source, conf=self.conf_floor, iou=self.nms_iou,
                                    imgsz=self.imgsz, max_det=self.max_det, verbose=False)[0]
        if self.duty:
            self.duty.end()                     # sleeps as long as the prediction took (duty 0.5)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return _EMPTY
        return (boxes.xyxy.cpu().numpy().astype(np.float32),
                boxes.conf.cpu().numpy().astype(np.float32),
                boxes.cls.cpu().numpy().astype(np.int32))


class SimulatedDetector:
    """Test detector with known behaviour (evaluation/simulate.py). Reads the
    ground truth of an image from the label file next to it
    (.../images/<role>/x.jpg -> .../labels/<role>/x.txt) and emits simulated
    candidates, deterministically per image.
    """

    def __init__(self, names: list[str], simulation: dict | None = None, seed: int = 0):
        from confdet.evaluation.simulate import SimConfig

        self.names = list(names)
        overrides = dict(simulation or {})
        # {dataset folder name: [detector class id for each of that dataset's class ids, or -1]}.
        # Lets the simulated detector be run on a dataset with a DIFFERENT taxonomy and
        # still answer in its own classes, as a real detector would.
        self.foreign_class_ids = {k: np.asarray(v, dtype=int)
                                  for k, v in (overrides.pop("foreign_class_ids", None) or {}).items()}
        n = len(self.names)
        if "class_probs" not in overrides:
            base = SimConfig().class_probs
            overrides["class_probs"] = (base + [0.02] * n)[:n]
        if "rare_classes" in overrides:
            overrides["rare_classes"] = tuple(overrides["rare_classes"])
        else:
            overrides["rare_classes"] = tuple(i for i, p in enumerate(overrides["class_probs"]) if p < 0.1)
        self.cfg = SimConfig(**overrides)
        self.seed = seed

    def predict(self, source):
        from confdet.data.occlusion import occlusion_severity_array
        from confdet.data.yolo_dataset import image_size, read_label_file
        from confdet.evaluation.simulate import simulate_detections, stable_seed

        image_path = Path(source)
        parts = list(image_path.parts)
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
        w, h = image_size(image_path)
        rows, _ = read_label_file(label_path, "bbox")
        gt_xyxy = np.array([[r[1] * w, r[2] * h, r[3] * w, r[4] * h] for r in rows], dtype=float).reshape(-1, 4)
        gt_cls = np.array([r[0] for r in rows], dtype=int)
        dataset = parts[idx - 1] if idx >= 1 else ""
        if dataset in self.foreign_class_ids and len(gt_cls):
            gt_cls = np.clip(self.foreign_class_ids[dataset][gt_cls], 0, None)   # unmapped -> class 0
        occ = occlusion_severity_array(gt_xyxy)
        cfg = dataclasses.replace(self.cfg, width=w, height=h)
        rng = np.random.default_rng(stable_seed(image_path.name, self.seed))
        xyxy, conf, cls = simulate_detections(gt_xyxy, gt_cls, occ, rng, cfg)
        return xyxy.astype(np.float32), conf.astype(np.float32), cls.astype(np.int32)


def load_detector(det: DetectorConfig, weights: str | Path, prediction_cfg: dict, names: list[str],
                  gpu_share: dict | None = None):
    """Detector ready to emit raw candidates.

    names: class names of the dataset the detector was trained on. For an
    Ultralytics checkpoint they are checked against the names stored in the
    checkpoint, so a checkpoint from a different dataset cannot be used by
    accident.
    """
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found -- run scripts/04_train_detector.py --detector {det.name}")
    if det.family == "simulated":
        meta = json.loads(weights.read_text())
        return SimulatedDetector(names, meta.get("simulation") or det.simulation, int(meta.get("seed", 0)))
    share = gpu_share or {}
    apply_memory_fraction(share.get("memory_fraction"))
    detector = UltralyticsDetector(
        weights, det.family, det.imgsz,
        float(prediction_cfg.get("conf_floor", 0.001)),
        float(prediction_cfg.get("nms_iou", 0.7)),
        int(prediction_cfg.get("max_det", 300)),
        duty_cycle=share.get("duty_cycle"),
    )
    if [str(n) for n in detector.names] != [str(n) for n in names]:
        raise ValueError(
            f"class names stored in {weights} ({detector.names}) differ from the development "
            f"dataset's ({names}). This checkpoint was trained on a different dataset or split; "
            f"retrain with scripts/04_train_detector.py.")
    return detector
