"""RQ5 (secondary, bounded) -- can post-hoc signals tell a miss caused by
occlusion from a miss caused by an unfamiliar vehicle type?
docs/theoretical_foundations.md section 7.5.

The plan phrases this through an aleatoric/epistemic split from an
evidential head, which needs a retrained detector. The question itself can
be put to the frozen detector directly, because the two causes leave
different fingerprints in its raw output. Every object missed at the
standard threshold falls into exactly one category:

  confused     a confident box of ANOTHER class sits on it, and that box
               belongs to no other vehicle: the detector saw this vehicle
               and misnamed it                      (familiarity failure)
  suppressed   its own, correctly classed candidate exists but scored below
               the threshold                        (visibility failure)
  merged       the only boxes on it belong to neighbouring vehicles: two
               vehicles, one box                    (visibility failure)
  unproposed   no box of any kind overlaps it       (visibility failure)
  weak_other   only weak, unattached boxes of other classes (ambiguous)

A box that is the matched detection of a neighbouring vehicle is never read
as a "confusion" of this one, however confident and whatever its class --
otherwise every heavily occluded vehicle sitting inside its occluder's box
would be miscounted as a naming error.

If the signals separate the causes, "confused" misses should follow class
rarity and not occlusion, and the visibility categories should follow
occlusion and not rarity. Two logistic regressions over ALL test objects
measure exactly that:

  P(visibility-type miss) ~ occlusion + rarity + size
  P(confusion miss)       ~ occlusion + rarity + size

rarity = -log(share of the class among detector-training objects). Odds
ratios are reported per +0.1 occlusion and per +1 SD rarity, with
image-bootstrap intervals. This is a descriptive, bounded analysis, labelled
as such; it is not a classifier and makes no claim beyond these associations.
"""
from __future__ import annotations

import numpy as np

from confdet.data.occlusion import bucket_indices
from confdet.evaluation.stats import logistic_irls, percentile_ci

CAUSES = ("confused", "suppressed", "merged", "unproposed", "weak_other")
VISIBILITY = ("suppressed", "merged", "unproposed")


def classify_misses(tables, std_threshold: float) -> np.ndarray:
    """Per ground-truth object: index into CAUSES, or -1 if it was found."""
    if tables.class_agnostic:
        raise ValueError("miss-cause analysis needs class-aware tables")
    found = tables.g_proposed & (tables.g_score >= std_threshold)
    has_box = tables.g_loc_cls >= 0                         # an own or unattached box overlaps it
    other_cls = has_box & (tables.g_loc_cls != tables.g_cls)
    confused = other_cls & (tables.g_loc_score >= std_threshold)
    cause = np.full(tables.n_gt, CAUSES.index("weak_other"), dtype=int)
    cause[~has_box & (tables.g_nbr_score > 0)] = CAUSES.index("merged")
    cause[~has_box & (tables.g_nbr_score <= 0)] = CAUSES.index("unproposed")
    cause[tables.g_proposed] = CAUSES.index("suppressed")
    cause[confused] = CAUSES.index("confused")
    cause[found] = -1
    return cause


def _odds_model(y, occ, rarity_z, area_z, img, boot_w, n_boot):
    n = len(y)
    if y.sum() < 10 or (n - y.sum()) < 10:
        return {"fitted": False, "reason": f"only {int(y.sum())} events"}
    X = np.column_stack([np.ones(n), occ * 10.0, rarity_z, area_z])
    beta = logistic_irls(X, y)
    reps = []
    for bw in boot_w[:n_boot]:
        w = bw[img]
        if (w * y).sum() < 3:
            continue
        reps.append(logistic_irls(X, y, weights=w)[1:3])
    reps = np.array(reps) if reps else np.zeros((0, 2))

    def ci(col):
        lo, hi = percentile_ci(reps[:, col]) if len(reps) else (np.nan, np.nan)
        return (float(np.exp(lo)), float(np.exp(hi))) if np.isfinite(lo) else (None, None)

    return {"fitted": True, "n_events": int(y.sum()),
            "odds_ratio_per_0.1_occlusion": float(np.exp(beta[1])), "occlusion_ci": ci(0),
            "odds_ratio_per_sd_rarity": float(np.exp(beta[2])), "rarity_ci": ci(1)}


def miss_cause_analysis(
    tables,
    std_threshold: float,
    train_class_counts: np.ndarray,
    bucket_edges: list[float],
    boot_w: np.ndarray,
    class_names: list[str] | None = None,
    n_boot: int = 200,
) -> dict:
    cause = classify_misses(tables, std_threshold)
    missed = cause >= 0
    n_missed = int(missed.sum())
    counts = np.asarray(train_class_counts, dtype=float)
    share = (counts + 1.0) / (counts.sum() + len(counts))
    rarity = -np.log(share)[np.clip(tables.g_cls, 0, len(share) - 1)]
    out = {
        "n_objects": int(tables.n_gt), "n_missed": n_missed, "standard_threshold": std_threshold,
        "cause_counts": {c: int((cause == i).sum()) for i, c in enumerate(CAUSES)},
        "cause_shares": {c: float((cause == i).sum() / n_missed) if n_missed else None
                         for i, c in enumerate(CAUSES)},
        "training_class_share": {(class_names[i] if class_names and i < len(class_names) else str(i)):
                                 float(counts[i] / counts.sum()) if counts.sum() else None
                                 for i in range(len(counts))},
    }
    if n_missed == 0:
        return out

    out["mean_occlusion_by_cause"] = {c: float(tables.g_occ[cause == i].mean()) if (cause == i).any() else None
                                      for i, c in enumerate(CAUSES)}
    out["mean_rarity_by_cause"] = {c: float(rarity[cause == i].mean()) if (cause == i).any() else None
                                   for i, c in enumerate(CAUSES)}
    b = bucket_indices(tables.g_occ, bucket_edges)
    out["cause_shares_by_occlusion_bucket"] = []
    for k in range(len(bucket_edges) - 1):
        m = missed & (b == k)
        row = {"bucket": k, "lo": bucket_edges[k], "hi": bucket_edges[k + 1], "n_missed": int(m.sum())}
        for i, c in enumerate(CAUSES):
            row[c] = float((cause[m] == i).mean()) if m.any() else None
        out["cause_shares_by_occlusion_bucket"].append(row)

    rarity_z = (rarity - rarity.mean()) / max(rarity.std(), 1e-9)
    area_z = (tables.g_area - tables.g_area.mean()) / max(tables.g_area.std(), 1e-9)
    vis = np.isin(cause, [CAUSES.index(c) for c in VISIBILITY]).astype(float)
    conf = (cause == CAUSES.index("confused")).astype(float)
    m_vis = _odds_model(vis, tables.g_occ, rarity_z, area_z, tables.g_img, boot_w, n_boot)
    m_conf = _odds_model(conf, tables.g_occ, rarity_z, area_z, tables.g_img, boot_w, n_boot)
    out["visibility_miss_model"] = m_vis
    out["confusion_miss_model"] = m_conf

    def excludes_one_above(ci):
        return ci[0] is not None and ci[0] > 1.0

    verdict = None
    if m_vis.get("fitted") and m_conf.get("fitted"):
        vis_occ = excludes_one_above(m_vis["occlusion_ci"])
        conf_rar = excludes_one_above(m_conf["rarity_ci"])
        conf_occ = excludes_one_above(m_conf["occlusion_ci"])
        vis_rar = excludes_one_above(m_vis["rarity_ci"])
        verdict = {
            "visibility_misses_rise_with_occlusion": vis_occ,
            "confusion_misses_rise_with_rarity": conf_rar,
            "confusion_misses_rise_with_occlusion": conf_occ,
            "visibility_misses_rise_with_rarity": vis_rar,
            "causes_separated": bool(vis_occ and conf_rar and not conf_occ and not vis_rar),
        }
    out["verdict"] = verdict
    return out
