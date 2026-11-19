"""RQ3 -- do prediction-only calibration metrics understate unreliability by
ignoring misses, and does a miss-aware metric change which detector looks
more reliable?   docs/theoretical_foundations.md sections 6 and 7.3.

Reported per detector:

  * D-ECE, MA-ECE, the gap, the miss mass phi, miss-aware Brier and OCE at
    the standard operating threshold, with image-bootstrap intervals;
  * the same quantities across a sweep of thresholds. D-ECE is free to fall
    as the threshold rises (fewer, surer detections; the vehicles dropped on
    the way are invisible to it), while MA-ECE and OCE pay for every drop.
    The threshold each metric would pick is reported -- "the metric rewards
    silence" stated as a number;
  * the exact split of the gap across occlusion buckets. Since
    gap = phi * (1 - D-ECE) and phi is a sum over missed objects, bucket b
    accounts for the fraction (misses in b) / (all misses) of the gap. Set
    against the bucket's share of objects, this says how over-represented
    occluded vehicles are in what D-ECE does not see.

Across detectors (scripts/08_compare_detectors.py): ranking under each
metric, and the bootstrap probability that the metrics disagree on order.
"""
from __future__ import annotations

import numpy as np

from confdet.calibration.detection_ece import (
    miss_aware_brier,
    miss_aware_ece,
    oce_per_object,
    operating_point,
    reliability_bins,
)
from confdet.data.occlusion import bucket_indices
from confdet.evaluation.stats import percentile_ci

METRICS = ("dece", "ma_ece", "ma_brier", "oce_max")


def metric_replicates(tables, std_threshold: float, num_bins: int, boot_w: np.ndarray) -> dict[str, np.ndarray]:
    """Bootstrap replicates of every metric at the standard threshold. With
    the same boot_w and the same test images, replicates of two detectors are
    PAIRED, which is what a ranking comparison needs.
    """
    acc = tables.c_conf >= std_threshold
    conf, tp, cimg = tables.c_conf[acc], tables.c_tp[acc], tables.c_img[acc]
    found = (tables.g_proposed & (tables.g_score >= std_threshold)).astype(float)
    n_img = tables.n_images
    gt_per_img = np.bincount(tables.g_img, minlength=n_img).astype(float)
    found_per_img = np.bincount(tables.g_img, weights=found, minlength=n_img)
    brier_obj = oce_per_object(tables, std_threshold, "max")
    oce_per_img = np.bincount(tables.g_img, weights=brier_obj, minlength=n_img)
    reps = {k: [] for k in ("dece", "ma_ece", "gap", "miss_mass", "ma_brier", "oce_max", "recall")}
    for bw in boot_w:
        n_gt = float(bw @ gt_per_img)
        n_missed = n_gt - float(bw @ found_per_img)
        w = bw[cimg]
        m = miss_aware_ece(conf, tp, n_missed, num_bins, w)
        for k in ("dece", "ma_ece", "gap", "miss_mass"):
            reps[k].append(m[k])
        reps["ma_brier"].append(miss_aware_brier(conf, tp, n_missed, w))
        reps["oce_max"].append(float(bw @ oce_per_img) / n_gt if n_gt > 0 else np.nan)
        reps["recall"].append(float(bw @ found_per_img) / n_gt if n_gt > 0 else np.nan)
    return {k: np.array(v) for k, v in reps.items()}


def calibration_analysis(
    tables,
    std_threshold: float,
    sweep_thresholds: list[float],
    num_bins: int,
    bucket_edges: list[float],
    boot_w: np.ndarray,
    min_bucket_n: int = 30,
) -> dict:
    point = operating_point(tables, std_threshold, num_bins)
    reps = metric_replicates(tables, std_threshold, num_bins, boot_w)
    for k in ("dece", "ma_ece", "gap", "miss_mass", "ma_brier", "oce_max"):
        point[f"{k}_ci"] = percentile_ci(reps[k])

    sweep = [operating_point(tables, float(t), num_bins) for t in sweep_thresholds]
    chosen = {}
    for metric in ("dece", "ma_ece", "ma_brier", "oce_max"):
        vals = [(row[metric], row["threshold"]) for row in sweep if row["n_detections"] > 0]
        if vals:
            best = min(vals)
            row = next(r for r in sweep if r["threshold"] == best[1])
            chosen[metric] = {"threshold": best[1], "value": best[0], "recall_there": row["recall"]}

    # exact decomposition of the gap over occlusion buckets
    found = tables.g_proposed & (tables.g_score >= std_threshold)
    missed = (~found).astype(float)
    b = bucket_indices(tables.g_occ, bucket_edges)
    n_gt, n_missed = tables.n_gt, float(missed.sum())
    n_img = tables.n_images
    decomposition = []
    for k in range(len(bucket_edges) - 1):
        m = b == k
        nk, mk = int(m.sum()), float(missed[m].sum())
        row = {"bucket": k, "lo": bucket_edges[k], "hi": bucket_edges[k + 1],
               "n_objects": nk, "n_missed": int(mk), "reliable": nk >= min_bucket_n,
               "share_of_objects": nk / n_gt if n_gt else None,
               "share_of_gap": mk / n_missed if n_missed else None,
               "miss_rate": mk / nk if nk else None}
        if nk and n_missed:
            num = np.bincount(tables.g_img[m], weights=missed[m], minlength=n_img)
            den = np.bincount(tables.g_img, weights=missed, minlength=n_img)
            with np.errstate(divide="ignore", invalid="ignore"):
                row["share_of_gap_ci"] = percentile_ci((boot_w @ num) / (boot_w @ den))
            row["gap_contribution"] = (mk / n_missed) * point["gap"]
            row["over_representation"] = (mk / n_missed) / (nk / n_gt) if nk else None
        decomposition.append(row)

    return {
        "at_standard_threshold": point,
        # MA-Brier = false-alarm term + matched-detection term + miss term (exact)
        "ma_brier_decomposition_at_standard": point["ma_brier_decomposition"],
        "threshold_sweep": sweep,
        "threshold_each_metric_prefers": chosen,
        "gap_decomposition_by_occlusion": decomposition,
        "reliability_bins": reliability_bins(tables, std_threshold, num_bins),
        "identity_check": abs(point["gap"] - point["miss_mass"] * (1 - point["dece"])) < 1e-9,
    }


def threshold_for_false_alarms(tables, fp_per_image: float) -> float:
    """Confidence threshold at which the detector emits (as nearly as its
    scores allow, never more than) `fp_per_image` false alarms per image.
    Used to compare detectors at MATCHED COST instead of at one shared
    confidence value, which means different things to different detectors.
    """
    fp_conf = np.sort(tables.c_conf[~tables.c_tp])[::-1]
    k = int(np.floor(fp_per_image * tables.n_images))
    if k <= 0:
        return 1.0
    if k >= len(fp_conf):
        return 0.0
    # accept confidences strictly above the (k+1)-th highest false-alarm confidence
    return float(np.nextafter(fp_conf[k], 1.0))


def _separation_sign(ci) -> int:
    """+1 / -1 when a paired-difference interval lies entirely above / below
    zero, 0 when it includes zero (or is undefined)."""
    lo, hi = ci
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return 0
    if lo > 0:
        return 1
    if hi < 0:
        return -1
    return 0


def ranking_comparison(per_detector: dict[str, dict]) -> dict:
    """per_detector: name -> {"point": {metric: value}, "reps": {metric: array}}
    (replicates paired across detectors). Lower is better for every metric.

    Per pair and metric, `separates` says whether the 95% paired-difference
    interval excludes zero. Two metrics FLIP on a pair only if both separate
    it and the signs disagree; a difference of point estimates inside the
    noise is recorded as `point_order_differs_vs_dece`, never as a flip.
    """
    names = sorted(per_detector)
    if len(names) < 2:
        return {"detectors": names, "note": "ranking needs at least two detectors"}
    out = {"detectors": names, "rankings": {}, "pairwise": []}
    for metric in METRICS:
        out["rankings"][metric] = sorted(names, key=lambda n: per_detector[n]["point"][metric])
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            row = {"a": a, "b": c}
            for metric in METRICS:
                pa, pb = per_detector[a]["point"][metric], per_detector[c]["point"][metric]
                d = per_detector[a]["reps"][metric] - per_detector[c]["reps"][metric]
                ci = percentile_ci(d)
                sign = _separation_sign(ci)
                row[metric] = {"a": pa, "b": pb, "better": a if pa < pb else c,
                               "difference_a_minus_b_ci": ci,
                               "prob_a_better": float(np.mean(d < 0)),
                               "separates": sign != 0,
                               "better_if_separated": (a if sign < 0 else c) if sign else None}
            sign_of = {m: _separation_sign(row[m]["difference_a_minus_b_ci"]) for m in METRICS}
            row["separated_by"] = [m for m in METRICS if sign_of[m] != 0]
            row["flips_vs_dece"] = {m: bool(sign_of["dece"] and sign_of[m] and sign_of[m] != sign_of["dece"])
                                    for m in METRICS if m != "dece"}
            row["point_order_differs_vs_dece"] = {m: row[m]["better"] != row["dece"]["better"]
                                                  for m in METRICS if m != "dece"}
            row["flips_between_metrics"] = [
                [m1, m2] for k, m1 in enumerate(METRICS) for m2 in METRICS[k + 1:]
                if sign_of[m1] and sign_of[m2] and sign_of[m1] != sign_of[m2]]
            # probability, over bootstrap resamples, that D-ECE and MA-ECE order this pair differently
            d1 = per_detector[a]["reps"]["dece"] - per_detector[c]["reps"]["dece"]
            d2 = per_detector[a]["reps"]["ma_ece"] - per_detector[c]["reps"]["ma_ece"]
            row["prob_dece_and_ma_ece_disagree"] = float(np.mean((d1 < 0) != (d2 < 0)))
            out["pairwise"].append(row)
    out["any_ranking_flip_vs_dece"] = any(any(r["flips_vs_dece"].values()) for r in out["pairwise"])
    out["pairs_dece_cannot_separate"] = [[r["a"], r["b"]] for r in out["pairwise"] if not r["dece"]["separates"]]
    return out
