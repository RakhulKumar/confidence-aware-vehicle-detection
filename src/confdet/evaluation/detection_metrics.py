"""Plain detection quality numbers, reported before anything else.

These are the gate every calibration statistic has to pass: if a detector
finds almost nothing on a dataset (as happened in the first cross-dataset
run, where class ids of two unrelated taxonomies were compared), every
downstream number is noise, and the summary must say so first.
"""
from __future__ import annotations

import numpy as np


def average_precision(conf: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """COCO-style 101-point interpolated AP from labelled candidates."""
    if n_gt <= 0:
        return float("nan")
    if len(conf) == 0:
        return 0.0
    order = np.argsort(-np.asarray(conf), kind="stable")
    tp = np.asarray(tp, dtype=float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    for i in range(len(precision) - 2, -1, -1):          # precision envelope
        precision[i] = max(precision[i], precision[i + 1])
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    vals = np.where(idx < len(precision), precision[np.clip(idx, 0, len(precision) - 1)], 0.0)
    return float(vals.mean())


def detection_summary(tables, std_threshold: float, class_names: list[str] | None = None) -> dict:
    n_gt = tables.n_gt
    found = tables.g_proposed & (tables.g_score >= std_threshold)
    acc = tables.c_conf >= std_threshold
    n_det = int(acc.sum())
    n_tp = int((acc & tables.c_tp).sum())
    out = {
        "n_images": tables.n_images, "n_ground_truth": int(n_gt), "n_candidates": int(tables.n_cand),
        "objects_per_image": n_gt / tables.n_images if tables.n_images else 0.0,
        "class_agnostic": bool(tables.class_agnostic),
        "recall_at_floor": float(tables.g_proposed.mean()) if n_gt else None,
        "recall_at_standard": float(found.mean()) if n_gt else None,
        "precision_at_standard": (n_tp / n_det) if n_det else None,
        "fp_per_image_at_standard": (n_det - n_tp) / tables.n_images if tables.n_images else None,
    }
    if tables.class_agnostic:
        out["ap50"] = average_precision(tables.c_conf, tables.c_tp, n_gt)
    else:
        per_class = {}
        classes = np.unique(tables.g_cls) if n_gt else np.zeros(0, dtype=int)
        for c in classes:
            cm, gm = tables.c_cls == c, tables.g_cls == c
            name = class_names[c] if class_names and c < len(class_names) else str(int(c))
            per_class[name] = {
                "n": int(gm.sum()),
                "ap50": average_precision(tables.c_conf[cm], tables.c_tp[cm], int(gm.sum())),
                "recall_at_standard": float(found[gm].mean()),
            }
        out["per_class"] = per_class
        aps = [v["ap50"] for v in per_class.values() if np.isfinite(v["ap50"])]
        out["ap50"] = float(np.mean(aps)) if aps else None
    return out


def usable_for_calibration_analysis(summary: dict, min_recall_at_floor: float = 0.5) -> tuple[bool, str]:
    """A dataset on which the detector proposes a candidate for fewer than
    half of the vehicles is not measuring calibration; it is measuring a
    detector that does not work there.
    """
    r = summary.get("recall_at_floor")
    if r is None:
        return False, "no ground-truth objects"
    if r < min_recall_at_floor:
        return False, (f"the detector proposes a candidate for only {r:.1%} of the vehicles "
                       f"(need >= {min_recall_at_floor:.0%}); check the class mapping and whether "
                       f"the detector transfers to this dataset at all before reading anything else")
    return True, ""
