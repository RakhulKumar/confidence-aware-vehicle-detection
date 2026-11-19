"""Controlled occlusion probe -- a diagnostic, never a substitute for real
evaluation data.

Design. Pick N annotated vehicles. For EACH of them build one image per
severity level s in {0, 0.1, ..., 0.9} by pasting the SAME occluder crop (a
real vehicle cut from another image) over the bottom s-fraction of its box.
The occluder keeps its aspect ratio, is at least as wide as the target and
hangs down below it, the way a nearer vehicle does on an oblique camera, so
the covered share of the target box is exactly s.

Because every target appears at every level with the same occluder, this is
a within-vehicle dose-response experiment: size, class, viewpoint, lighting
and background are held fixed while occlusion alone is varied. That is the
one thing the observational analysis (RQ1 on real data) cannot do, and it is
the only occlusion evidence available on a dataset whose images contain
almost no real vehicle-on-vehicle occlusion.

Limits, stated: pasted rectangles have unnatural edges, and a pasted vehicle
may itself be detected. Results are written to their own clearly named file
and must be reported as a controlled probe.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from confdet.data.yolo_dataset import read_label_file
from confdet.evaluation.stats import percentile_ci


@dataclass
class ProbeTarget:
    image_path: Path
    box: tuple[float, float, float, float]      # x1, y1, x2, y2 pixels
    class_id: int
    occluder_path: Path
    occluder_box: tuple[float, float, float, float]


def _boxes_of(image_path: Path, label_path: Path):
    with Image.open(image_path) as im:
        w, h = im.size
    rows, _ = read_label_file(label_path, "bbox")
    return [(c, x1 * w, y1 * h, x2 * w, y2 * h) for c, x1, y1, x2, y2 in rows]


def select_targets(pairs: list[tuple[Path, Path]], n_targets: int, min_side: int = 32,
                   seed: int = 42, isolated_only: bool = True) -> list[ProbeTarget]:
    """Choose target vehicles and give each a fixed occluder from another image.

    isolated_only: use targets that no other annotated box overlaps, so the
    constructed severity is the vehicle's ONLY occlusion.
    """
    rng = random.Random(seed)
    candidates, occluders = [], []
    for image_path, label_path in pairs:
        boxes = _boxes_of(image_path, label_path)
        for i, (c, x1, y1, x2, y2) in enumerate(boxes):
            if (x2 - x1) < min_side or (y2 - y1) < min_side:
                continue
            occluders.append((image_path, (x1, y1, x2, y2)))
            if isolated_only and any(
                    j != i and min(x2, b[3]) > max(x1, b[1]) and min(y2, b[4]) > max(y1, b[2])
                    for j, b in enumerate(boxes)):
                continue
            candidates.append((image_path, (x1, y1, x2, y2), c))
    if not candidates or len(occluders) < 2:
        raise ValueError("not enough annotated vehicles of usable size to build a probe")
    rng.shuffle(candidates)
    targets = []
    for image_path, box, c in candidates[:n_targets]:
        others = [o for o in occluders if o[0] != image_path] or occluders
        occ_path, occ_box = rng.choice(others)
        targets.append(ProbeTarget(image_path, box, c, occ_path, occ_box))
    return targets


def paste_occluder(image: Image.Image, box, occluder: Image.Image, severity: float
                   ) -> tuple[Image.Image, tuple[float, float, float, float] | None]:
    """New image with `occluder` covering exactly the bottom `severity`
    fraction of `box` (full box width), extending below the box.

    Returns (image, occluder_box): occluder_box is the pasted crop's
    (x1, y1, x2, y2) clipped to the image, or None at severity 0. The probe
    scorer needs it, because the pasted crop is itself a vehicle and a
    detection on it must not be credited to the target it hides.
    """
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    out = image.copy()
    if severity == 0.0:
        return out, None
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    strip_h = bh * severity
    ow, oh = occluder.size
    scale = max(bw / ow, strip_h / oh)                 # at least as wide and as tall as the strip
    new_w, new_h = max(1, int(np.ceil(ow * scale))), max(1, int(np.ceil(oh * scale)))
    resized = occluder.resize((new_w, new_h))
    left = int(round((x1 + x2) / 2 - new_w / 2))
    left = min(left, int(np.floor(x1)))                # keep the full box width covered
    if left + new_w < x2:
        left = int(np.ceil(x2)) - new_w
    top = int(round(y2 - strip_h))
    out.paste(resized, (left, top))
    W, H = out.size
    occ_box = (float(max(0, left)), float(max(0, top)), float(min(W, left + new_w)), float(min(H, top + new_h)))
    return out, occ_box


def covered_fraction_of_box(image_size, box, occluder_size, severity: float) -> float:
    """Geometric check used by the tests: share of `box` the paste covers."""
    if severity == 0.0:
        return 0.0
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    strip_h = bh * severity
    ow, oh = occluder_size
    scale = max(bw / ow, strip_h / oh)
    new_w, new_h = int(np.ceil(ow * scale)), int(np.ceil(oh * scale))
    left = min(int(round((x1 + x2) / 2 - new_w / 2)), int(np.floor(x1)))
    if left + new_w < x2:
        left = int(np.ceil(x2)) - new_w
    top = int(round(y2 - strip_h))
    ix = max(0.0, min(x2, left + new_w) - max(x1, left))
    iy = max(0.0, min(y2, top + new_h, image_size[1]) - max(y1, top))
    return ix * iy / (bw * bh)


def render_probe_images(target: ProbeTarget, severities: list[float]
                        ) -> list[tuple[Image.Image, tuple[float, float, float, float] | None]]:
    """One (image, occluder_box) per severity; see paste_occluder."""
    with Image.open(target.image_path) as im:
        base = im.convert("RGB")
    with Image.open(target.occluder_path) as im:
        ox1, oy1, ox2, oy2 = target.occluder_box
        occluder = im.convert("RGB").crop((int(ox1), int(oy1), int(np.ceil(ox2)), int(np.ceil(oy2))))
    return [paste_occluder(base, target.box, occluder, s) for s in severities]


def analyse_probe(scores: np.ndarray, severities: list[float], std_threshold: float,
                  n_boot: int = 1000, seed: int = 0) -> dict:
    """scores: (n_targets, n_severities) object score of each target at each
    level (0 where the detector proposed nothing). Targets are the
    resampling unit.

    Only targets the detector finds when unoccluded are analysed -- a vehicle
    it cannot find at all says nothing about the effect of occluding it.
    """
    scores = np.asarray(scores, dtype=float)
    n_all = len(scores)
    base_found = scores[:, 0] >= std_threshold
    s = scores[base_found]
    n = len(s)
    out = {"n_targets_built": int(n_all), "n_targets": int(n), "standard_threshold": std_threshold,
           "found_when_unoccluded": float(base_found.mean()) if n_all else None, "by_severity": []}
    if n < 5:
        out["note"] = "too few targets detected at severity 0 to analyse"
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    missed, unc = (s < std_threshold).astype(float), 1.0 - s
    for j, sev in enumerate(severities):
        out["by_severity"].append({
            "severity": float(sev),
            "miss_rate": float(missed[:, j].mean()), "miss_rate_ci": percentile_ci(missed[idx, j].mean(axis=1)),
            "mean_uncertainty": float(unc[:, j].mean()), "mean_uncertainty_ci": percentile_ci(unc[idx, j].mean(axis=1)),
        })
    sev = np.asarray(severities, dtype=float)
    centred = sev - sev.mean()
    slopes = (unc * centred).sum(axis=1) / (centred ** 2).sum()       # per-target slope of uncertainty on severity
    out["mean_within_target_slope"] = float(slopes.mean())
    out["slope_ci"] = percentile_ci(slopes[idx].mean(axis=1))
    out["share_of_targets_with_positive_slope"] = float((slopes > 0).mean())
    out["dose_response_monotone"] = bool(all(
        a["miss_rate"] <= b["miss_rate"] + 0.02 for a, b in zip(out["by_severity"], out["by_severity"][1:])))
    return out


def probe_predictions(predict, targets: list[ProbeTarget], severities: list[float]) -> list[list[dict]]:
    """Run the detector once on every (target, severity) image.

    predict: callable(HxWx3 BGR uint8 array) -> (xyxy, confidence, class).
    Returns [target][severity] -> {"xyxy", "conf", "cls", "occluder_box"}.
    Scoring is separate (score_probe), so one detector pass serves every
    matching mode.
    """
    out = []
    for t in targets:
        row = []
        for image, occ_box in render_probe_images(t, severities):
            bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
            xyxy, conf, cls = predict(bgr)
            row.append({"xyxy": np.asarray(xyxy, dtype=float).reshape(-1, 4),
                        "conf": np.asarray(conf, dtype=float).reshape(-1),
                        "cls": np.asarray(cls).reshape(-1), "occluder_box": occ_box})
        out.append(row)
    return out


def score_probe(preds: list[list[dict]], targets: list[ProbeTarget], iou_threshold: float = 0.5,
                class_aware: bool = True) -> np.ndarray:
    """(n_targets, n_severities): the highest confidence among candidates that
    count for the target, 0.0 when none does. A candidate counts if
      IoU(candidate, target) >= iou_threshold, AND
      IoU(candidate, target) >  IoU(candidate, pasted occluder box)
    (and it has the target's class, if class_aware). The second condition
    keeps a box drawn on the pasted vehicle -- which hangs over the target
    and can overlap it by more than the IoU threshold -- from being credited
    as finding the vehicle it hides.
    """
    from confdet.data.occlusion import iou_matrix

    out = np.zeros((len(targets), len(preds[0]) if preds else 0))
    for i, t in enumerate(targets):
        box = np.array([t.box], dtype=float)
        for j, p in enumerate(preds[i]):
            if len(p["conf"]) == 0:
                continue
            iou_t = iou_matrix(p["xyxy"], box)[:, 0]
            iou_o = (iou_matrix(p["xyxy"], np.array([p["occluder_box"]], dtype=float))[:, 0]
                     if p["occluder_box"] is not None else np.zeros(len(iou_t)))
            ok = (iou_t >= iou_threshold) & (iou_t > iou_o)
            if class_aware:
                ok &= p["cls"] == t.class_id
            if ok.any():
                out[i, j] = float(p["conf"][ok].max())
    return out


def probe_scores(predict, targets: list[ProbeTarget], severities: list[float],
                 iou_threshold: float = 0.5, class_aware: bool = True) -> np.ndarray:
    """probe_predictions followed by score_probe."""
    return score_probe(probe_predictions(predict, targets, severities), targets, iou_threshold, class_aware)
