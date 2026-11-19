"""Detector wrapper, training plumbing, post-hoc fitting and the probe
scorer. Ultralytics and torch are replaced by small fakes, so these tests
check THIS project's code around those libraries: the arguments it passes,
the checkpoint it keeps, the guard it installs, the arrays it returns.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.evaluation.labeling import build_tables  # noqa: E402
from confdet.evaluation.simulate import simulate_records  # noqa: E402
from confdet.models import detector_factory as df  # noqa: E402
from confdet.pipeline import PostHoc, fit_posthoc  # noqa: E402

CFG = {
    "experiment": {"name": "t", "development": "d"},
    "posthoc": {"alpha": 0.2, "alpha_grid": [0.1, 0.3], "context_conf": 0.25,
                "context_edges": [0.0, 0.05, 0.3, 1.0], "min_group_size": 100, "calibrator_fraction": 0.5},
    "evaluation": {"standard_operating_threshold": 0.25},
    "occlusion": {"proxy": "bottom_y"},
    "detectors": {"active": "a", "models": {
        "a": {"family": "yolo", "weights": "yolo26m.pt", "batch": 8},
        "old": {"family": "ultralytics", "weights": "rtdetr-l.pt"},
        "bad": {"family": "yolo", "learning_rate": 0.1},
        "sim": {"family": "simulated"}}},
}


class _Tensor:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __len__(self):
        return len(self.a)


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _Tensor(xyxy), _Tensor(conf), _Tensor(cls)

    def __len__(self):
        return len(self.conf)


class _FakeModel:
    """Stands in for ultralytics.YOLO / RTDETR."""
    instances = []

    def __init__(self, weights):
        self.weights, self.callbacks, self.train_kwargs, self.predict_kwargs = weights, {}, None, None
        self.names = {0: "car", 1: "bus"}
        self.trainer = None
        _FakeModel.instances.append(self)

    def add_callback(self, event, fn):
        self.callbacks.setdefault(event, []).append(fn)       # Ultralytics appends, it never replaces

    def _fire(self, event):
        for fn in self.callbacks.get(event, []):
            fn(self.trainer)

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        if kwargs.get("resume"):
            # resume=True: Ultralytics restores everything from the checkpoint it was built from
            save_dir = Path(self.weights).parent.parent
        else:
            # Ultralytics decides the run folder itself; emulate it landing somewhere unexpected
            save_dir = Path(kwargs["project"]) / "train7"
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"BEST")
        self.trainer = types.SimpleNamespace(best=str(save_dir / "weights" / "best.pt"), save_dir=str(save_dir))
        for _ in range(12):
            self._fire("on_train_batch_start")
            self._fire("on_train_batch_end")

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        boxes = _Boxes([[0, 0, 10, 10], [5, 5, 20, 20]], [0.9, 0.002], [1.0, 0.0])
        return [types.SimpleNamespace(boxes=boxes)]


def _install_fakes(reserved_gb=6.0, total_gb=16.0):
    _FakeModel.instances.clear()
    fake_ul = types.ModuleType("ultralytics")
    fake_ul.YOLO = type("YOLO", (_FakeModel,), {})
    fake_ul.RTDETR = type("RTDETR", (_FakeModel,), {})
    cuda = types.SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 0, fraction_calls=[],
        set_per_process_memory_fraction=lambda f, dev: cuda.fraction_calls.append((f, dev)),
        memory_reserved=lambda dev: int(reserved_gb * 2 ** 30),
        get_device_properties=lambda dev: types.SimpleNamespace(total_memory=int(total_gb * 2 ** 30)))
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = cuda
    # remember the real modules (if already imported) so _remove_fakes can put them back: deleting a
    # real `torch` from sys.modules makes the next `import torch` re-initialise it and crash
    _SAVED.update({k: sys.modules.get(k) for k in ("ultralytics", "torch")})
    sys.modules["ultralytics"], sys.modules["torch"] = fake_ul, fake_torch


_SAVED: dict = {}


def _remove_fakes():
    for k in ("ultralytics", "torch"):
        if _SAVED.get(k) is not None:
            sys.modules[k] = _SAVED[k]
        else:
            sys.modules.pop(k, None)
    _SAVED.clear()


def test_detector_config_parsing():
    assert df.detector_config(CFG).name == "a" and df.detector_config(CFG).batch == 8
    assert df.detector_config(CFG, "old").family == "rtdetr"          # old "ultralytics" value still works
    with pytest.raises(ValueError, match="train_args"):
        df.detector_config(CFG, "bad")
    with pytest.raises(KeyError):
        df.detector_config(CFG, "missing")


def test_training_keeps_one_canonical_checkpoint_and_passes_safe_arguments(tmp_path):
    _install_fakes()
    try:
        det = df.detector_config(CFG, "a")
        best = df.train(det, tmp_path / "data.yaml", tmp_path / "run", device="cuda", seed=3)
        model = _FakeModel.instances[-1]
        kw = model.train_kwargs
        assert best == (tmp_path / "run" / "weights" / "best.pt").resolve() and best.read_bytes() == b"BEST"
        assert Path(kw["project"]).is_absolute() and Path(kw["data"]).is_absolute()
        assert kw["exist_ok"] is True and kw["batch"] == 8 and kw["seed"] == 3
        assert "optimizer" not in kw and "lr0" not in kw            # optimizer auto: lr0 would be ignored anyway
        assert type(model).__name__ == "YOLO"

        rt = df.DetectorConfig(name="r", family="rtdetr", weights="rtdetr-l.pt", optimizer="AdamW", lr0=1e-4,
                               train_args={"cos_lr": True})
        df.train(rt, tmp_path / "data.yaml", tmp_path / "run_r")
        kw = _FakeModel.instances[-1].train_kwargs
        assert type(_FakeModel.instances[-1]).__name__ == "RTDETR"
        assert kw["optimizer"] == "AdamW" and kw["lr0"] == 1e-4 and kw["cos_lr"] is True
    finally:
        _remove_fakes()


def test_train_args_seed_overrides_the_project_seed(tmp_path):
    _install_fakes()
    try:
        det = df.DetectorConfig(name="a_s1", family="yolo", weights="yolo26m.pt", train_args={"seed": 1})
        df.train(det, tmp_path / "data.yaml", tmp_path / "run", seed=42)
        assert _FakeModel.instances[-1].train_kwargs["seed"] == 1
        df.train(df.DetectorConfig(name="a", family="yolo"), tmp_path / "data.yaml", tmp_path / "run0", seed=42)
        assert _FakeModel.instances[-1].train_kwargs["seed"] == 42
    finally:
        _remove_fakes()
    sim = df.train(df.DetectorConfig(name="s", family="simulated", train_args={"seed": 7}),
                   tmp_path / "d.yaml", tmp_path / "sim", seed=42)
    assert json.loads(sim.read_text())["seed"] == 7
    # the experiment config's seed variants differ from their base detector only in train_args.seed
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml").read_text("utf-8"))
    models = cfg["detectors"]["models"]
    for base in ("yolo26", "rtdetr"):
        for s in (1, 2):
            v = dict(models[f"{base}_s{s}"])
            assert v.pop("train_args") == {"seed": s} and v == models[base]
    assert "seed" not in models["yolo26"].get("train_args", {}) and cfg["project"]["seed"] == 42


def test_duty_cycle_pauses_as_long_as_the_work_took():
    t = {"now": 0.0}
    slept = []
    dc = df.DutyCycle(0.5, sleeper=lambda s: slept.append(s), clock=lambda: t["now"])
    for busy in (0.2, 0.5, 0.1):
        dc.start()
        t["now"] += busy
        dc.end()
    assert slept == pytest.approx([0.2, 0.5, 0.1])
    assert dc.stats["busy_s"] == pytest.approx(0.8) and dc.stats["slept_s"] == pytest.approx(0.8)
    assert df.DutyCycle(0.25, sleeper=lambda s: None).ratio == pytest.approx(3.0)       # busy 1/4 of the time
    dc.end()                                                                          # end without start: no-op
    assert dc.stats["units"] == 3
    with pytest.raises(ValueError):
        df.DutyCycle(0.0)
    assert set(dc.callbacks()) == {"on_train_batch_start", "on_train_batch_end", "on_val_batch_start",
                                   "on_val_batch_end"}


def test_training_under_a_gpu_share_caps_vram_and_throttles(tmp_path):
    _install_fakes()
    try:
        det = df.detector_config(CFG, "a")
        df.train(det, tmp_path / "d.yaml", tmp_path / "run", gpu_share={"memory_fraction": 0.45, "duty_cycle": 0.5})
        m = _FakeModel.instances[-1]
        assert sys.modules["torch"].cuda.fraction_calls == [(0.45, 0)]
        assert len(m.callbacks["on_train_batch_end"]) == 2                   # VRAM guard + duty cycle
        assert {"on_val_batch_start", "on_val_batch_end"} <= set(m.callbacks)
    finally:
        _remove_fakes()
    _install_fakes(reserved_gb=7.5, total_gb=16.0)                         # 7.5 GB is above a 0.45 share
    try:
        with pytest.raises(df.VRAMOverflowError, match="limit of 7.2 GB"):
            df.train(df.detector_config(CFG, "a"), tmp_path / "d.yaml", tmp_path / "run2",
                     gpu_share={"memory_fraction": 0.45})
    finally:
        _remove_fakes()


def test_resume_continues_from_last_checkpoint_and_never_restarts(tmp_path):
    _install_fakes()
    try:
        det = df.DetectorConfig(name="a_s1", family="yolo", weights="yolo26m.pt", train_args={"seed": 1})
        run = tmp_path / "run"
        last = df.last_checkpoint(run)
        last.parent.mkdir(parents=True)
        last.write_bytes(b"LAST")                                          # an interrupted run
        best = df.train(det, tmp_path / "d.yaml", run, resume=True)
        m = _FakeModel.instances[-1]
        assert Path(m.weights) == last and m.train_kwargs == {"resume": True}   # nothing re-specified
        assert best.read_bytes() == b"BEST" and not list(best.parent.glob("*.tmp"))
        # without resume, or without a last.pt, a fresh run starts from the pretrained weights
        df.train(det, tmp_path / "d.yaml", tmp_path / "fresh", resume=True)
        assert _FakeModel.instances[-1].weights == "yolo26m.pt" and _FakeModel.instances[-1].train_kwargs["seed"] == 1
    finally:
        _remove_fakes()
    # a finished run (Ultralytics stores epoch -1) is not retrained
    import torch
    run2 = tmp_path / "done"
    last2 = df.last_checkpoint(run2)
    last2.parent.mkdir(parents=True)
    torch.save({"epoch": -1}, last2)
    (last2.parent / "best.pt").write_bytes(b"FINAL")
    n_before = len(_FakeModel.instances)
    out = df.train(df.DetectorConfig(name="x", family="yolo"), tmp_path / "d.yaml", run2, resume=True)
    assert out.read_bytes() == b"FINAL" and len(_FakeModel.instances) == n_before


def test_runs_dir_moves_training_outputs_and_writes_are_atomic(tmp_path):
    from confdet.evaluation.records import RecordSet, load_records, save_records
    from confdet.utils.io_utils import load_results, save_results
    from confdet.utils.paths import detector_dir, train_dir, weights_path
    cfg = {"project": {"output_dir": str(tmp_path / "out")}, "experiment": {"name": "e"}}
    assert train_dir(cfg, "d") == detector_dir(cfg, "d") and weights_path(cfg, "d").parent.parent == detector_dir(cfg, "d")
    cfg["project"]["runs_dir"] = str(tmp_path / "runs")
    assert train_dir(cfg, "d") == tmp_path / "runs" / "e" / "d"
    assert weights_path(cfg, "d") == tmp_path / "runs" / "e" / "d" / "weights" / "best.pt"
    save_results({"a": 1}, tmp_path / "r" / "x.json")
    assert load_results(tmp_path / "r" / "x.json") == {"a": 1} and not list((tmp_path / "r").glob("*.tmp"))
    rs = RecordSet(simulate_records(3, seed=1), {"dataset": "d", "role": "test"})
    save_records(tmp_path / "p" / "d__test.npz", rs)
    assert [p.name for p in (tmp_path / "p").iterdir()] == ["d__test.npz"]
    assert len(load_records(tmp_path / "p" / "d__test.npz").records) == 3


def test_prediction_file_counts_as_done_only_when_complete(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("predict05", Path(__file__).resolve().parents[1] / "scripts" / "05_predict.py")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from confdet.evaluation.records import RecordSet, save_records
    recs = simulate_records(4, seed=2)
    pairs = [(Path(f"x/{r.image_id}"), None) for r in recs]
    path = tmp_path / "d__test.npz"
    assert not mod.prediction_file_complete(path, pairs)
    save_records(path, RecordSet(recs[:3], {}))
    assert not mod.prediction_file_complete(path, pairs)                     # partial -> redo
    save_records(path, RecordSet(recs, {}))
    assert mod.prediction_file_complete(path, pairs)
    path.write_bytes(b"garbage")
    assert not mod.prediction_file_complete(path, pairs)                     # unreadable -> redo


def test_occlusion_report_refuses_missing_development_data_and_writes_nothing(tmp_path):
    import subprocess
    import yaml
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "configs" / "experiment.yaml").read_text("utf-8"))
    cfg["project"].update({"processed_dir": str(tmp_path / "empty"), "output_dir": str(tmp_path / "out")})
    (tmp_path / "empty").mkdir()
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(repo / "scripts" / "03_compute_occlusion.py"), "--config", str(path)],
                          capture_output=True, text=True, cwd=repo)
    assert proc.returncode != 0 and "Nothing was written" in proc.stderr
    assert not list((tmp_path / "out").rglob("occlusion_report.json"))


def test_fit_posthoc_uses_the_configured_score_features():
    import copy
    calib = build_tables(simulate_records(200, seed=4))
    fitted = fit_posthoc(CFG, calib, seed=0)                         # no posthoc.score_features -> full_strict
    assert fitted["calibrator"]["name"] == "full_strict" and fitted["meta"]["score_features"] == "full_strict"
    assert "dupS" in fitted["calibrator"]["features"]
    cfg = copy.deepcopy(CFG)
    cfg["posthoc"]["score_features"] = "occlusion_aware"
    assert fit_posthoc(cfg, calib, seed=0)["calibrator"]["features"] == ["z", "ctx", "dup", "z*ctx", "z*dup"]
    assert PostHoc(fitted).calibrator.name == "full_strict"


def test_vram_guard_stops_training_that_overflows_the_card(tmp_path):
    _install_fakes(reserved_gb=20.5, total_gb=16.0)                   # the numbers from the first real run
    try:
        with pytest.raises(df.VRAMOverflowError, match="Lower `batch`"):
            df.train(df.detector_config(CFG, "a"), tmp_path / "d.yaml", tmp_path / "run")
        warnings = []
        logger = types.SimpleNamespace(warning=warnings.append, info=lambda *_: None)
        df.train(df.detector_config(CFG, "a"), tmp_path / "d.yaml", tmp_path / "run2",
                 abort_on_vram_overflow=False, logger=logger)
        assert len(warnings) == 1 and "20.5 GB" in warnings[0]
    finally:
        _remove_fakes()
    _install_fakes(reserved_gb=15.5, total_gb=16.0)                   # full but fitting: no interference
    try:
        df.train(df.detector_config(CFG, "a"), tmp_path / "d.yaml", tmp_path / "run3")
    finally:
        _remove_fakes()


def test_ultralytics_wrapper_returns_raw_candidates_and_checks_class_names(tmp_path):
    _install_fakes()
    try:
        w = tmp_path / "best.pt"
        w.write_bytes(b"x")
        pred_cfg = {"conf_floor": 0.001, "nms_iou": 0.7, "max_det": 300}
        det = df.load_detector(df.detector_config(CFG, "a"), w, pred_cfg, ["car", "bus"])
        xyxy, conf, cls = det.predict(tmp_path / "img.jpg")
        kw = _FakeModel.instances[-1].predict_kwargs
        assert kw["conf"] == 0.001 and kw["imgsz"] == 960 and kw["max_det"] == 300 and isinstance(kw["source"], str)
        assert xyxy.shape == (2, 4) and conf.dtype == np.float32 and cls.tolist() == [1, 0]
        det.predict(np.zeros((4, 4, 3), dtype=np.uint8))
        assert isinstance(_FakeModel.instances[-1].predict_kwargs["source"], np.ndarray)
        with pytest.raises(ValueError, match="different dataset"):
            df.load_detector(df.detector_config(CFG, "a"), w, pred_cfg, ["Hatchback", "Sedan"])
        with pytest.raises(FileNotFoundError, match="04_train_detector"):
            df.load_detector(df.detector_config(CFG, "a"), tmp_path / "nope.pt", pred_cfg, ["car", "bus"])
    finally:
        _remove_fakes()


def test_simulated_detector_is_deterministic_and_reads_adjacent_labels(tmp_path):
    from PIL import Image

    img = tmp_path / "ds" / "images" / "test" / "a.jpg"
    lbl = tmp_path / "ds" / "labels" / "test" / "a.txt"
    img.parent.mkdir(parents=True); lbl.parent.mkdir(parents=True)
    Image.new("RGB", (640, 360)).save(img)
    lbl.write_text("0 0.3 0.6 0.2 0.3\n1 0.6 0.7 0.25 0.3\n2 0.5 0.5 0.1 0.1")
    det_cfg = df.detector_config(CFG, "sim")
    w = df.train(det_cfg, tmp_path / "x.yaml", tmp_path / "run", seed=5)
    assert json.loads(w.read_text())["family"] == "simulated"
    d = df.load_detector(det_cfg, w, {}, ["car", "rickshaw", "bike"])
    a, b = d.predict(img), d.predict(img)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and len(a[1]) >= 1
    assert a[0][:, [0, 2]].max() <= 640 and a[0][:, [1, 3]].max() <= 360


def test_posthoc_fit_uses_disjoint_images_for_score_and_thresholds_and_round_trips():
    calib = build_tables(simulate_records(400, seed=31))
    fitted = fit_posthoc(CFG, calib, seed=1)
    m = fitted["meta"]
    assert m["n_images_for_score"] + m["n_images_for_thresholds"] == calib.n_images
    assert 0.35 < m["n_images_for_score"] / calib.n_images < 0.65
    assert set(fitted["thresholds"]) == {"0.1", "0.2", "0.3"}
    ph = PostHoc(json.loads(json.dumps(fitted)))                      # survives JSON
    assert ph.alpha == 0.2 and set(ph.rules()) == {"global_confidence", "mondrian_confidence",
                                                   "global_occlusion_aware"}
    test = build_tables(simulate_records(100, seed=32))
    inputs = ph.rule_inputs(test)
    assert all(len(score) == test.n_cand for _, score in inputs.values())
    # a stricter target can only lower (or keep) the threshold
    t = lambda a: ph.rules(a)["global_confidence"].groups[0].threshold  # noqa: E731
    assert t(0.1) <= t(0.2) <= t(0.3)


def test_posthoc_fit_survives_a_calib_set_too_small_for_the_score():
    tiny = build_tables(simulate_records(4, seed=1))
    fitted = fit_posthoc(CFG, tiny, seed=0)
    assert fitted["calibrator"] is None and fitted["notes"]
    assert "global_occlusion_aware" not in fitted["thresholds"]["0.2"]
    assert PostHoc(fitted).calibrator is None


def test_probe_scores_follow_what_the_detector_sees(tmp_path):
    from PIL import Image
    from confdet.data.synthetic_occlusion import ProbeTarget, analyse_probe, probe_scores

    base, occ = tmp_path / "base.jpg", tmp_path / "occ.png"
    Image.new("RGB", (400, 300), (0, 0, 0)).save(base)
    Image.new("RGB", (60, 60), (255, 255, 255)).save(occ)
    box = (100.0, 60.0, 220.0, 180.0)
    targets = [ProbeTarget(base, box, 1, occ, (0, 0, 60, 60)) for _ in range(6)]

    def predict(bgr):
        """A 'detector' whose confidence is the visible share of the target box."""
        visible = 1.0 - (bgr[60:180, 100:220, 0] > 127).mean()
        return np.array([box]), np.array([0.95 * visible]), np.array([1])

    sev = [0.0, 0.3, 0.6, 0.9]
    scores = probe_scores(predict, targets, sev)
    assert scores.shape == (6, 4)
    assert np.allclose(scores[0], [0.95 * (1 - s) for s in sev], atol=0.03)
    r = analyse_probe(scores, sev, 0.25)
    assert r["by_severity"][0]["miss_rate"] == 0.0 and r["by_severity"][-1]["miss_rate"] == 1.0
    wrong_class = probe_scores(lambda b: (np.array([box]), np.array([0.9]), np.array([0])), targets, sev)
    assert (wrong_class == 0).all()


def _white_box(bgr):
    ys, xs = np.nonzero(bgr[:, :, 0] > 127)
    return None if len(xs) == 0 else [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def test_probe_does_not_credit_a_detection_on_the_pasted_occluder(tmp_path):
    from PIL import Image
    from confdet.data.occlusion import iou_matrix
    from confdet.data.synthetic_occlusion import ProbeTarget, probe_predictions, score_probe

    base, occ = tmp_path / "base.jpg", tmp_path / "occ.png"
    Image.new("RGB", (400, 300), (0, 0, 0)).save(base)
    Image.new("RGB", (60, 60), (255, 255, 255)).save(occ)
    box = (100.0, 60.0, 220.0, 180.0)
    targets = [ProbeTarget(base, box, 1, occ, (0, 0, 60, 60))]
    sev = [0.0, 0.5, 0.9]

    def only_occluder(bgr):
        """A 'detector' that boxes the pasted vehicle and nothing else."""
        b = _white_box(bgr)
        return (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)) if b is None else \
            (np.array([b]), np.array([0.9]), np.array([1]))

    preds = probe_predictions(only_occluder, targets, sev)
    # at 0.9 the occluder box overlaps the target by more than 0.5 -- the old rule would have counted it
    occ_box = preds[0][2]["occluder_box"]
    assert iou_matrix(np.array([occ_box]), np.array([box]))[0, 0] >= 0.5
    for class_aware in (True, False):
        assert (score_probe(preds, targets, 0.5, class_aware) == 0).all()

    def both(bgr):
        """The occluder (confident) and the target itself (less so)."""
        b = _white_box(bgr)
        boxes, conf = [list(box)], [0.7]
        if b is not None:
            boxes.append(b); conf.append(0.9)
        return np.array(boxes), np.array(conf), np.ones(len(conf), int)

    scores = score_probe(probe_predictions(both, targets, sev), targets, 0.5, class_aware=False)
    assert np.allclose(scores, 0.7)


# ---------------------------------------------------------------------------
# matching modes and per-detector operating thresholds (added after the first real run)
# ---------------------------------------------------------------------------
def _mode_cfg(threshold="max_f1"):
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["experiment"] = {"name": "t", "development": "dev", "transfer": ["other"]}
    cfg["evaluation"].update({"standard_operating_threshold": threshold,
                              "common_classes": ["car", "two_wheeler", "bus"],
                              "matching_modes": ["class_agnostic", "native"]})
    cfg["posthoc"]["context_conf"] = None
    cfg["datasets"] = {
        "dev": {"class_map": {"Hatchback": "car", "Sedan": "car", "Bike": "two_wheeler", "Bus": "bus", "Cycle": None}},
        "other": {"class_map": {"Car": "car", "Scooter": "two_wheeler", "Coach": "bus"}},
    }
    return cfg


def _mode_records():
    from confdet.evaluation.records import ImageRecord, RecordSet
    # one Sedan detected as a Hatchback (misnamed), one Bike found, one Bus with nothing on it,
    # one Cycle (no counterpart in the common taxonomy) found
    rec = ImageRecord("a.jpg", 1000, 1000,
                      gt_xyxy=[[0, 0, 100, 100], [200, 200, 300, 300], [500, 500, 700, 700], [800, 800, 900, 900]],
                      gt_cls=[1, 2, 3, 4], gt_occ=[0, 0, 0, 0],
                      cand_xyxy=[[0, 0, 100, 100], [200, 200, 300, 300], [800, 800, 900, 900]],
                      cand_conf=[0.9, 0.8, 0.7], cand_cls=[0, 2, 4])
    names = ["Hatchback", "Sedan", "Bike", "Bus", "Cycle"]
    return RecordSet([rec], {"dataset": "dev", "role": "test", "dataset_names": names, "detector_names": names})


def test_matching_modes_change_what_counts_as_a_miss():
    from confdet.pipeline import tables_from_records

    cfg, rs = _mode_cfg(0.25), _mode_records()
    native = tables_from_records(cfg, rs, "native")
    coarse = tables_from_records(cfg, rs, "coarse")
    agnostic = tables_from_records(cfg, rs, "class_agnostic")
    assert native.g_proposed.tolist() == [False, True, False, True]       # the misnamed Sedan is a "miss"
    assert coarse.n_gt == 3 and coarse.g_proposed.tolist() == [True, True, False]   # Sedan/Hatchback both "car"; Cycle ignored
    assert coarse.n_cand == 2                                             # the box on the ignored Cycle is dropped
    assert agnostic.g_proposed.tolist() == [True, True, False, True]      # only the Bus is silently missed
    with pytest.raises(ValueError):
        tables_from_records(cfg, rs, "strict")


def test_another_dataset_is_never_matched_by_raw_class_index():
    from confdet.evaluation.records import ImageRecord, RecordSet
    from confdet.pipeline import tables_from_records, transfer_mode

    cfg = _mode_cfg(0.25)
    # dataset "other": class 0 = Car. The detector answers with ITS class ids: 1 = Sedan.
    rec = ImageRecord("b.jpg", 1000, 1000, [[0, 0, 100, 100]], [0], [0], [[0, 0, 100, 100]], [0.9], [1])
    rs = RecordSet([rec], {"dataset": "other", "role": "test", "dataset_names": ["Car", "Scooter", "Coach"],
                           "detector_names": ["Hatchback", "Sedan", "Bike", "Bus", "Cycle"]})
    assert transfer_mode("native") == "coarse" and transfer_mode("class_agnostic") == "class_agnostic"
    for mode in ("native", "coarse", "class_agnostic"):                   # "native" falls back to the common taxonomy
        assert tables_from_records(cfg, rs, mode).g_proposed.tolist() == [True]


def test_max_f1_threshold_and_calibration_tables():
    from confdet.evaluation.records import RecordSet
    from confdet.pipeline import calibration_tables, max_f1_threshold

    recs = simulate_records(150, seed=41)
    names = ["a", "b", "c", "d", "e"]
    rs = RecordSet(recs, {"dataset": "dev", "role": "calib", "dataset_names": names, "detector_names": names})
    cfg = _mode_cfg("max_f1")
    T, tau, ctx = calibration_tables(cfg, rs, "class_agnostic")
    assert 0.05 <= tau <= 0.95 and ctx == tau and T.context_conf == tau

    def f1(t):
        k = (T.c_conf >= t).sum()
        return 2 * (T.c_tp & (T.c_conf >= t)).sum() / (k + T.n_gt)
    best, value = max_f1_threshold(T)
    assert best == tau and value == pytest.approx(f1(tau))
    assert all(f1(tau) >= f1(t) - 1e-12 for t in np.round(np.arange(0.05, 0.951, 0.01), 2))

    cfg_fixed = _mode_cfg(0.4)
    _, tau2, ctx2 = calibration_tables(cfg_fixed, rs, "native")
    assert tau2 == 0.4 and ctx2 == 0.4
    fitted = fit_posthoc(cfg, T, seed=0, tau=tau, mode="class_agnostic")
    ph = PostHoc(fitted)
    assert ph.tau == tau and ph.context_conf == tau and ph.mode == "class_agnostic"


def test_config_validation_of_modes_and_threshold(tmp_path):
    import yaml
    from confdet.utils.io_utils import load_config, matching_modes

    base = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml").read_text())
    assert matching_modes(base)[0] == "class_agnostic"
    for key, bad in (("matching_modes", ["class_agnostic", "fuzzy"]), ("standard_operating_threshold", "best"),
                     ("standard_operating_threshold", 1.5)):
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["evaluation"][key] = bad
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError):
            load_config(p)


def test_train_script_never_retrains_a_finished_cleaned_run_unless_asked(tmp_path):
    import subprocess
    import yaml
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "configs" / "experiment.yaml").read_text("utf-8"))
    cfg["project"].update({"processed_dir": str(tmp_path / "proc"), "output_dir": str(tmp_path / "out"),
                           "runs_dir": str(tmp_path / "runs")})
    cfg["experiment"] = {"name": "e", "development": "bmd45", "transfer": []}
    cfg["detectors"]["models"] = {"sim": {"family": "simulated"}}
    cfg["detectors"]["active"] = "sim"
    for role in ("fit", "val"):
        (tmp_path / "proc" / "e" / "bmd45" / "images" / role).mkdir(parents=True)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    best = tmp_path / "runs" / "e" / "sim" / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_text("FINISHED")                          # a finished run whose last.pt was cleaned up
    run = lambda *a: subprocess.run([sys.executable, str(repo / "scripts" / "04_train_detector.py"), "--config",  # noqa: E731
                                     str(path), *a], capture_output=True, text=True, cwd=repo)
    r = run("--resume")
    assert r.returncode == 0 and "nothing to do" in (r.stdout + r.stderr) and best.read_text() == "FINISHED"
    r = run("--resume", "--retrain")
    assert r.returncode == 0 and best.read_text() != "FINISHED"   # explicitly asked: trained again


def test_evidential_family_trains_through_its_own_trainer_and_resumes(tmp_path):
    from confdet.models import evidential_yolo
    sentinel = object()
    orig = evidential_yolo.evidential_trainer
    evidential_yolo.evidential_trainer = lambda: sentinel
    _install_fakes()
    try:
        det = df.DetectorConfig(name="ev", family="yolo_evidential", weights="yolo26m.pt", batch=4)
        df.train(det, tmp_path / "d.yaml", tmp_path / "run")
        m = _FakeModel.instances[-1]
        assert type(m).__name__ == "YOLO" and m.train_kwargs["trainer"] is sentinel
        assert evidential_yolo.epoch_callback in m.callbacks["on_train_epoch_start"]
        last = df.last_checkpoint(tmp_path / "run2")
        last.parent.mkdir(parents=True)
        last.write_bytes(b"L")
        df.train(det, tmp_path / "d.yaml", tmp_path / "run2", resume=True)
        assert _FakeModel.instances[-1].train_kwargs == {"resume": True, "trainer": sentinel}
    finally:
        _remove_fakes()
        evidential_yolo.evidential_trainer = orig


def test_runs_dir_keeps_legacy_weights_where_they_are(tmp_path):
    from confdet.utils.paths import train_dir, weights_path
    cfg = {"project": {"output_dir": str(tmp_path / "out"), "runs_dir": str(tmp_path / "runs")},
           "experiment": {"name": "e"}}
    old = tmp_path / "out" / "e" / "old" / "weights" / "best.pt"
    old.parent.mkdir(parents=True)
    old.write_text("x")
    assert weights_path(cfg, "old") == old                                   # trained before runs_dir: found in place
    assert train_dir(cfg, "new") == tmp_path / "runs" / "e" / "new"         # anything new goes to runs_dir
