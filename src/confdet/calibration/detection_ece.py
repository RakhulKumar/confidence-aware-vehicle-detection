"""Detection calibration metrics.  docs/theoretical_foundations.md section 6.

Three metrics, each blind to something different:

  D-ECE    (Kueppers et al., 2020)  bins the EMITTED detections by confidence
           and compares confidence with precision. A vehicle the detector
           says nothing about never enters the sum: silence is free.

  OCE      (Park, Sobolewski & Azizan, arXiv:2412.01782)  averages a Brier
           score over GROUND-TRUTH OBJECTS; an object with no overlapping
           detection scores the maximum error 1. It sees misses, but a false
           alarm on empty road overlaps no object and never enters the sum.

  MA-ECE   (this project)  D-ECE with the misses put back in. Every emitted
           detection is a claim "a vehicle is here" with stated probability =
           confidence; every missed vehicle is the same claim with stated
           probability 0 and outcome 1. Pooling both populations gives

               MA-ECE = (N_det * D-ECE + N_miss) / (N_det + N_miss)
                      = (1 - phi) * D-ECE + phi,       phi = N_miss / (N_det + N_miss)

           so  MA-ECE - D-ECE = phi * (1 - D-ECE) >= 0, with equality iff
           nothing is missed. The gap is driven entirely by the miss mass phi,
           and phi splits exactly across occlusion buckets, which is what
           lets RQ3 say how much of the understatement occlusion accounts for.

A correction to the earlier version of this file
------------------------------------------------
The earlier "object-level ECE with raw candidates" set an object's outcome to
"did its best raw candidate survive the operating threshold". That outcome is
a deterministic function of the confidence itself (1 if confidence >= tau,
else 0), so the quantity measured how far confidences sit from a step
function, not calibration, and it scored a detector that proposed nothing for
an object (confidence 0, outcome 0) as perfectly calibrated on it. It was also
attributed to the OCE paper, which defines something different. The outcome
of a ground-truth object in a calibration metric is "the object exists" (1),
never "the detector kept it".

All functions take flat arrays (one entry per detection) and optional
per-entry weights, so image-level bootstrap replicates are weighted sums.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from confdet.data.occlusion import Box


# ---------------------------------------------------------------------------
# Small object types used by inference / visualisation code
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float
    class_id: int
    id: int = -1


@dataclass(frozen=True)
class GroundTruth:
    box: Box
    class_id: int
    occlusion_severity: float = 0.0


def match_detections_to_ground_truth(
    detections: list[Detection],
    ground_truths: list[GroundTruth],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> tuple[list[Detection | None], list[bool]]:
    """List-based convenience wrapper around labeling.greedy_match for ONE
    image. Returns (matched detection per ground truth or None, is-true-
    positive per detection).
    """
    from confdet.data.occlusion import boxes_to_array
    from confdet.evaluation.labeling import greedy_match

    c2g, g2c = greedy_match(
        boxes_to_array(d.box for d in detections),
        np.array([d.confidence for d in detections], dtype=float),
        np.array([d.class_id for d in detections], dtype=int),
        boxes_to_array(g.box for g in ground_truths),
        np.array([g.class_id for g in ground_truths], dtype=int),
        iou_threshold, class_agnostic,
    )
    return [detections[k] if k >= 0 else None for k in g2c], [bool(j >= 0) for j in c2g]


# ---------------------------------------------------------------------------
# Binned calibration error
# ---------------------------------------------------------------------------
def binned_calibration_error(confidences, outcomes, num_bins: int = 10, weights=None
                             ) -> tuple[float, list[dict]]:
    """sum_m (W_m / W) * |mean outcome_m - mean confidence_m| over equal-width
    confidence bins. Returns (error, per-bin rows). 0.0 for an empty input.
    """
    conf = np.asarray(confidences, dtype=float)
    out = np.asarray(outcomes, dtype=float)
    w = np.ones(len(conf)) if weights is None else np.asarray(weights, dtype=float)
    total = float(w.sum())
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    if len(conf) == 0 or total <= 0:
        return 0.0, []
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, num_bins - 1)
    wsum = np.bincount(idx, weights=w, minlength=num_bins)
    csum = np.bincount(idx, weights=w * conf, minlength=num_bins)
    osum = np.bincount(idx, weights=w * out, minlength=num_bins)
    error, rows = 0.0, []
    for m in range(num_bins):
        if wsum[m] <= 0:
            rows.append({"lo": float(edges[m]), "hi": float(edges[m + 1]), "weight": 0.0,
                         "accuracy": None, "confidence": None})
            continue
        acc, cbar = osum[m] / wsum[m], csum[m] / wsum[m]
        error += (wsum[m] / total) * abs(acc - cbar)
        rows.append({"lo": float(edges[m]), "hi": float(edges[m + 1]), "weight": float(wsum[m]),
                     "accuracy": float(acc), "confidence": float(cbar)})
    return float(error), rows


def dece(confidences, is_tp, num_bins: int = 10, weights=None) -> float:
    """Standard Detection-ECE over the detections passed in (callers pass
    the detections at or above the operating threshold).
    """
    return binned_calibration_error(confidences, is_tp, num_bins, weights)[0]


def miss_aware_ece(confidences, is_tp, n_missed: float, num_bins: int = 10, weights=None) -> dict:
    """MA-ECE and its decomposition.

    confidences / is_tp / weights: the emitted detections.
    n_missed: number (or total weight) of ground-truth objects with no true-
    positive detection among them.
    """
    w = np.ones(len(np.asarray(confidences))) if weights is None else np.asarray(weights, dtype=float)
    n_det = float(w.sum())
    d = dece(confidences, is_tp, num_bins, weights)
    denom = n_det + float(n_missed)
    if denom <= 0:
        return {"dece": 0.0, "ma_ece": 0.0, "miss_mass": 0.0, "gap": 0.0}
    phi = float(n_missed) / denom
    ma = (1 - phi) * d + phi
    return {"dece": float(d), "ma_ece": float(ma), "miss_mass": float(phi), "gap": float(ma - d)}


def miss_aware_brier(confidences, is_tp, n_missed: float, weights=None) -> float:
    """Proper-scoring counterpart of MA-ECE: mean squared error of the claim
    "a vehicle is here" over emitted detections (outcome = is_tp) and missed
    objects (stated probability 0, outcome 1).
    """
    conf = np.asarray(confidences, dtype=float)
    tp = np.asarray(is_tp, dtype=float)
    w = np.ones(len(conf)) if weights is None else np.asarray(weights, dtype=float)
    denom = float(w.sum()) + float(n_missed)
    if denom <= 0:
        return 0.0
    return float((np.sum(w * (tp - conf) ** 2) + float(n_missed)) / denom)


def miss_aware_brier_decomposition(confidences, is_tp, n_missed: float, weights=None) -> dict:
    """Exact split of MA-Brier into three additive terms (they sum to MA-Brier):

        false_alarm   sum over false-positive detections of conf^2       / D
        matched       sum over true-positive detections of (1 - conf)^2  / D
        miss          number of missed objects (each costs 1)            / D
        D = number of emitted detections + number of missed objects

    `share_*` = term / MA-Brier. OCE has no false-alarm term at all (a box on
    empty road overlaps no object), so a detector whose MA-Brier is driven by
    false alarms can rank better under OCE than under MA-Brier.
    """
    conf = np.asarray(confidences, dtype=float)
    tp = np.asarray(is_tp, dtype=bool)
    w = np.ones(len(conf)) if weights is None else np.asarray(weights, dtype=float)
    denom = float(w.sum()) + float(n_missed)
    if denom <= 0:
        return {"ma_brier": 0.0, "false_alarm": 0.0, "matched": 0.0, "miss": 0.0,
                "share_false_alarm": None, "share_matched": None, "share_miss": None}
    fa = float(np.sum(w[~tp] * conf[~tp] ** 2)) / denom
    mt = float(np.sum(w[tp] * (1 - conf[tp]) ** 2)) / denom
    ms = float(n_missed) / denom
    total = fa + mt + ms
    share = (lambda x: x / total) if total > 0 else (lambda x: None)
    return {"ma_brier": total, "false_alarm": fa, "matched": mt, "miss": ms,
            "share_false_alarm": share(fa), "share_matched": share(mt), "share_miss": share(ms),
            "n_false_alarms": float(np.sum(w[~tp])), "n_matched": float(np.sum(w[tp])),
            "n_missed": float(n_missed)}


# ---------------------------------------------------------------------------
# OCE (Park et al.), top-1 approximation
# ---------------------------------------------------------------------------
def oce_per_object(tables, threshold: float, variant: str = "max") -> np.ndarray:
    """Per-ground-truth-object Brier score of Object-level Calibration Error
    with the detections kept at `threshold`, following arXiv:2412.01782
    eq. (6)-(10):

        Brier_j = sum_c (1[c = label_j] - p_bar_j[c])^2

    where p_bar_j is the class-probability vector of the kept detection that
    overlaps object j best ("max"), or the mean over all kept detections
    overlapping it with IoU >= delta ("ens"); an object with no overlapping
    kept detection scores 1.

    Approximation: Ultralytics exposes one class and one score per box, not a
    probability vector, so each detection's vector is taken as `confidence`
    on its predicted class and 0 elsewhere. delta is the tables' IoU
    threshold (the paper averages delta = 0.5 and 0.75).

    Class-agnostic tables (tables.class_agnostic) have one class, "vehicle":
    the kept box is a claim "a vehicle is here" with probability = its
    confidence, whatever fine class it carries. "max" scores (1 - conf)^2 for
    the best-overlapping kept box, "ens" scores (1 - mean conf)^2 over the
    kept boxes overlapping the object. A misnamed box therefore costs nothing
    extra, as class-agnostic matching intends.
    """
    if variant not in ("max", "ens"):
        raise ValueError("variant must be 'max' or 'ens'")
    n_gt = tables.n_gt
    brier = np.ones(n_gt)
    if n_gt == 0:
        return brier
    gt_of_row = np.repeat(np.arange(n_gt), np.diff(tables.ov_ptr))
    keep = tables.ov_conf >= threshold
    g, conf, cls, iou = gt_of_row[keep], tables.ov_conf[keep], tables.ov_cls[keep], tables.ov_iou[keep]
    if len(g) == 0:
        return brier
    agnostic = bool(getattr(tables, "class_agnostic", False))
    if variant == "max":
        order = np.lexsort((-iou, g))                     # per object, best-overlapping first
        g_s, first = np.unique(g[order], return_index=True)
        sel = order[first]
        if agnostic:
            brier[g_s] = (1 - conf[sel]) ** 2
        else:
            right = cls[sel] == tables.g_cls[g_s]
            brier[g_s] = np.where(right, (1 - conf[sel]) ** 2, 1.0 + conf[sel] ** 2)
    elif agnostic:
        count = np.bincount(g, minlength=n_gt).astype(float)
        total = np.bincount(g, weights=conf, minlength=n_gt)
        has = count > 0
        brier[has] = (1 - total[has] / count[has]) ** 2
    else:
        count = np.bincount(g, minlength=n_gt).astype(float)
        n_cls = int(max(cls.max(), tables.g_cls.max())) + 1
        sums = np.bincount(g * n_cls + cls, weights=conf, minlength=n_gt * n_cls).reshape(n_gt, n_cls)
        has = count > 0
        pbar = sums[has] / count[has][:, None]
        lab = tables.g_cls[has]
        p_true = pbar[np.arange(len(lab)), lab]
        brier[has] = (1 - p_true) ** 2 + (pbar ** 2).sum(axis=1) - p_true ** 2
    return brier


def oce_top1(tables, threshold: float, variant: str = "max", weights_per_gt=None) -> float:
    """OCE = (weighted) mean of `oce_per_object`. 0.0 when there are no objects."""
    if tables.n_gt == 0:
        return 0.0
    brier = oce_per_object(tables, threshold, variant)
    w = np.ones(tables.n_gt) if weights_per_gt is None else np.asarray(weights_per_gt, dtype=float)
    total = float(w.sum())
    return float((w * brier).sum() / total) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Operating-point summaries from labelled tables
# ---------------------------------------------------------------------------
def operating_point(tables, threshold: float, num_bins: int = 10, image_weights=None,
                    with_oce: bool = True) -> dict:
    """Every calibration/detection number at one confidence threshold."""
    wi = np.ones(tables.n_images) if image_weights is None else np.asarray(image_weights, dtype=float)
    wc, wg = wi[tables.c_img], wi[tables.g_img]
    acc = tables.c_conf >= threshold
    found = tables.g_proposed & (tables.g_score >= threshold)
    n_gt, n_found = float(wg.sum()), float((wg * found).sum())
    n_det = float(wc[acc].sum())
    n_tp = float((wc * (acc & tables.c_tp)).sum())
    n_missed = n_gt - n_found
    out = miss_aware_ece(tables.c_conf[acc], tables.c_tp[acc], n_missed, num_bins, wc[acc])
    out.update({
        "threshold": float(threshold),
        "ma_brier": miss_aware_brier(tables.c_conf[acc], tables.c_tp[acc], n_missed, wc[acc]),
        "ma_brier_decomposition": miss_aware_brier_decomposition(tables.c_conf[acc], tables.c_tp[acc],
                                                                 n_missed, wc[acc]),
        "recall": n_found / n_gt if n_gt > 0 else None,
        "precision": n_tp / n_det if n_det > 0 else None,
        "fp_per_image": (n_det - n_tp) / float(wi.sum()) if wi.sum() > 0 else None,
        "n_detections": n_det, "n_ground_truth": n_gt, "n_missed": n_missed,
    })
    if with_oce:
        out["oce_max"] = oce_top1(tables, threshold, "max", wg)
        out["oce_ens"] = oce_top1(tables, threshold, "ens", wg)
    return out


def reliability_bins(tables, threshold: float, num_bins: int = 10) -> list[dict]:
    acc = tables.c_conf >= threshold
    return binned_calibration_error(tables.c_conf[acc], tables.c_tp[acc], num_bins)[1]
