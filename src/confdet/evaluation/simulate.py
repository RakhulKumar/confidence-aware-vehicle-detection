"""A simulated traffic scene generator and a simulated detector whose
behaviour is known exactly.

This is test equipment, not data. It exists so that every statistic in the
pipeline can be checked against a ground truth that is true by construction:

  * the detector's confidence falls with occlusion at a known rate;
  * a known share of occluded vehicles is never proposed at all;
  * rare classes are misnamed at a known rate;
  * weak true candidates sit behind confidently detected vehicles, while weak
    false candidates sit on open road or on top of an already detected
    vehicle -- so occlusion context genuinely carries information
    (`context_informative=True`), or does not (`False`).

Used by tests/ and by scripts/00_selftest.py. Nothing produced here may ever
be written to outputs/ as a result.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from confdet.data.occlusion import occlusion_severity_array
from confdet.evaluation.records import ImageRecord


@dataclass
class SimConfig:
    width: int = 1280
    height: int = 720
    mean_objects: float = 11.0
    class_probs: list[float] = field(default_factory=lambda: [0.40, 0.30, 0.20, 0.07, 0.03])
    rare_classes: tuple[int, ...] = (3, 4)
    # detector behaviour
    base_logit: float = 2.8             # confidence logit of a fully visible vehicle
    occlusion_slope: float = 5.0        # logit lost per unit of occlusion severity
    logit_noise: float = 1.0
    p_unproposed_base: float = 0.01
    p_unproposed_occluded: float = 0.30  # extra, times severity^2
    rare_confusion: float = 0.35        # P(rare-class vehicle is proposed under a wrong class)
    background_fp_rate: float = 2.5     # weak false candidates per image on open road
    duplicate_rate: float = 0.5         # weak second boxes per proposed vehicle
    context_informative: bool = True    # False: confidence unrelated to occlusion

    @property
    def n_classes(self) -> int:
        return len(self.class_probs)


def simulate_scene(rng: np.random.Generator, cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth boxes (n, 4) and classes (n,) of one oblique-camera frame:
    vehicles lower in the frame are nearer and larger, and overlap by chance.
    """
    n = int(rng.poisson(cfg.mean_objects))
    W, H = cfg.width, cfg.height
    yb = rng.uniform(0.30 * H, 0.98 * H, n)
    scale = 0.30 + 0.70 * (yb / H)
    w = 0.13 * W * scale * rng.uniform(0.8, 1.3, n)
    h = w * rng.uniform(0.65, 0.95, n)
    cx = rng.uniform(0.08 * W, 0.92 * W, n)
    xyxy = np.column_stack([cx - w / 2, yb - h, cx + w / 2, yb])
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, W)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, H)
    cls = rng.choice(cfg.n_classes, size=n, p=np.asarray(cfg.class_probs) / np.sum(cfg.class_probs))
    return xyxy.astype(float), cls.astype(int)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate_detections(gt_xyxy: np.ndarray, gt_cls: np.ndarray, gt_occ: np.ndarray,
                        rng: np.random.Generator, cfg: SimConfig
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw candidates (xyxy, confidence, class) the simulated detector emits."""
    W, H = cfg.width, cfg.height
    boxes, confs, classes = [], [], []
    n = len(gt_cls)
    common = [c for c in range(cfg.n_classes) if c not in cfg.rare_classes]
    for j in range(n):
        o = float(gt_occ[j])
        if rng.random() < cfg.p_unproposed_base + cfg.p_unproposed_occluded * o * o:
            continue                                            # silent miss: nothing proposed
        x1, y1, x2, y2 = gt_xyxy[j]
        bw, bh = x2 - x1, y2 - y1
        jit = rng.normal(0, 0.04, 4) * np.array([bw, bh, bw, bh])
        box = np.array([x1, y1, x2, y2]) + jit
        slope = cfg.occlusion_slope if cfg.context_informative else 0.0
        conf = _sigmoid(cfg.base_logit - slope * o + rng.normal(0, cfg.logit_noise))
        c = int(gt_cls[j])
        if c in cfg.rare_classes and rng.random() < cfg.rare_confusion:
            c = int(rng.choice(common))                         # seen, but misnamed
        boxes.append(box); confs.append(conf); classes.append(c)
        if rng.random() < cfg.duplicate_rate:                   # weak second box on the same vehicle
            shift = rng.uniform(-0.12, 0.12, 4) * np.array([bw, bh, bw, bh])
            boxes.append(np.array([x1, y1, x2, y2]) + shift)
            confs.append(_sigmoid(rng.normal(-2.6, 0.8)))
            classes.append(int(gt_cls[j]) if rng.random() < 0.6 else int(rng.integers(cfg.n_classes)))
    for _ in range(int(rng.poisson(cfg.background_fp_rate))):   # weak boxes on open road
        w = rng.uniform(0.04, 0.10) * W
        h = w * rng.uniform(0.6, 1.0)
        cx, cy = rng.uniform(w / 2, W - w / 2), rng.uniform(h / 2, 0.28 * H)
        boxes.append(np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
        confs.append(_sigmoid(rng.normal(-2.2, 1.0)))
        classes.append(int(rng.integers(cfg.n_classes)))
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    xyxy = np.array(boxes, dtype=float)
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, W)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, H)
    conf = np.clip(np.array(confs, dtype=float), 0.0011, 0.9999)
    return xyxy, conf, np.array(classes, dtype=int)


def simulate_records(n_images: int, seed: int = 0, cfg: SimConfig | None = None,
                     prefix: str = "sim") -> list[ImageRecord]:
    """Scenes + detections for `n_images` independent frames."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_images):
        gt_xyxy, gt_cls = simulate_scene(rng, cfg)
        occ = occlusion_severity_array(gt_xyxy)
        c_xyxy, c_conf, c_cls = simulate_detections(gt_xyxy, gt_cls, occ, rng, cfg)
        records.append(ImageRecord(f"{prefix}_{i:06d}.jpg", cfg.width, cfg.height,
                                   gt_xyxy, gt_cls, occ, c_xyxy, c_conf, c_cls))
    return records


def stable_seed(text: str, base: int = 0) -> int:
    """Deterministic per-image seed (Python's hash() is salted per process)."""
    return (zlib.crc32(text.encode("utf-8")) + 7919 * int(base)) % (2 ** 32)
