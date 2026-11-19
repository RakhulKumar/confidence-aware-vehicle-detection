"""RQ1 -- does the detector's post-hoc uncertainty track occlusion?
docs/theoretical_foundations.md section 7.1.

Per ground-truth object j:
    occlusion severity  o_j   (from annotations)
    uncertainty         u_j = 1 - S_j, with u_j = 1 if nothing was proposed
    missed              m_j = 1 if S_j < standard threshold (or unproposed)

What is reported, and why each piece is there:

  * n per occlusion bucket, first. In the first run 278 of 286 test objects
    had severity < 0.2 and two buckets held one object each; no statistic can
    be read off that, and the output now refuses to pretend otherwise.
  * miss rate and mean uncertainty per bucket, with image-bootstrap intervals.
  * Spearman rho(o, u), and the PARTIAL rho given object size and class.
    Under the bottom-edge depth proxy an occluded vehicle is also, on
    average, a farther and smaller one, and small objects get lower
    confidence for reasons unrelated to occlusion. The partial correlation
    and the regression below are what separate the two.
  * logistic regression  missed ~ occlusion + log(size) + class, reported as
    the odds ratio of a miss per +0.1 occlusion severity.
  * rho(o, o_hat): how well the deployable occlusion context tracks the
    annotated severity. The conformal strata are only as good as this.
"""
from __future__ import annotations

import numpy as np

from confdet.data.occlusion import bucket_indices
from confdet.evaluation.stats import (
    bootstrap_mean_ci,
    logistic_irls,
    partial_spearman,
    percentile_ci,
    resample_rows_by_image,
    spearman,
)


def _class_dummies(cls: np.ndarray, min_count: int = 10) -> np.ndarray:
    values, counts = np.unique(cls, return_counts=True)
    keep = values[counts >= min_count]
    if len(keep) <= 1:
        return np.zeros((len(cls), 0))
    ref = keep[np.argmax(counts[counts >= min_count])]       # most frequent class = reference
    cols = [(cls == v).astype(float) for v in keep if v != ref]
    return np.column_stack(cols) if cols else np.zeros((len(cls), 0))


def power_check(occ: np.ndarray, occluded_threshold: float, min_occluded: int) -> dict:
    n_occ = int(np.sum(np.asarray(occ) >= occluded_threshold))
    ok = n_occ >= min_occluded
    return {
        "n_objects": int(len(occ)), "n_occluded": n_occ,
        "occluded_threshold": occluded_threshold, "min_occluded_required": min_occluded,
        "adequate": bool(ok),
        "message": "" if ok else (
            f"only {n_occ} objects with occlusion >= {occluded_threshold} (need >= {min_occluded}): "
            f"occlusion-stratified results on this split are NOT interpretable"),
    }


SIZE_LABELS = ("small", "medium", "large")


def size_tertile_table(tables, std_threshold: float, bucket_edges: list[float], boot_w: np.ndarray,
                       min_bucket_n: int = 30) -> dict:
    """Miss rate by object-size tertile, and by occlusion bucket x size tertile.

    Size is the annotated box's area relative to the image (tables.g_area =
    log of that ratio). Tertile edges are the 1/3 and 2/3 quantiles over the
    objects analysed, so each tertile holds about a third of them. Under the
    bottom-edge depth proxy occluded vehicles are on average farther and
    smaller; this table shows whether the occlusion gradient survives inside
    each size band, which the logistic model only states as one number.
    """
    area = np.asarray(tables.g_area, dtype=float)
    n = len(area)
    if n == 0:
        return {"size_tertile_edges_log_rel_area": [], "by_size_tertile": [], "by_occlusion_and_size": []}
    found = tables.g_proposed & (tables.g_score >= std_threshold)
    missed = (~found).astype(float)
    img = tables.g_img
    q1, q2 = (float(v) for v in np.quantile(area, [1 / 3, 2 / 3]))
    tert = np.where(area < q1, 0, np.where(area < q2, 1, 2))
    b = bucket_indices(tables.g_occ, bucket_edges)

    def cell(m):
        nk = int(m.sum())
        row = {"n": nk, "reliable": nk >= min_bucket_n}
        if nk:
            row["miss_rate"] = float(missed[m].mean())
            row["miss_rate_ci"] = bootstrap_mean_ci(missed[m], img[m], boot_w)
        else:
            row["miss_rate"], row["miss_rate_ci"] = None, None
        return row

    lo_hi = [(None, q1), (q1, q2), (q2, None)]                 # None = open end
    by_size = []
    for k, label in enumerate(SIZE_LABELS):
        lo, hi = lo_hi[k]
        row = {"tertile": k, "size": label, "log_rel_area_range": [lo, hi],
               "rel_area_range": [None if lo is None else float(np.exp(lo)),
                                  None if hi is None else float(np.exp(hi))]}
        row.update(cell(tert == k))
        by_size.append(row)
    cross = []
    for j in range(len(bucket_edges) - 1):
        for k, label in enumerate(SIZE_LABELS):
            row = {"bucket": j, "lo": bucket_edges[j], "hi": bucket_edges[j + 1], "tertile": k, "size": label}
            row.update(cell((b == j) & (tert == k)))
            cross.append(row)
    return {"size_tertile_edges_log_rel_area": [q1, q2],
            "size_tertile_edges_rel_area": [float(np.exp(q1)), float(np.exp(q2))],
            "by_size_tertile": by_size, "by_occlusion_and_size": cross}


def occlusion_analysis(
    tables,
    std_threshold: float,
    bucket_edges: list[float],
    boot_w: np.ndarray,
    occluded_threshold: float = 0.2,
    min_occluded: int = 200,
    min_bucket_n: int = 30,
    n_rank_boot: int = 300,
    n_logit_boot: int = 200,
    seed: int = 0,
) -> dict:
    occ, img = tables.g_occ, tables.g_img
    n = len(occ)
    power = power_check(occ, occluded_threshold, min_occluded)
    if n == 0:
        return {"power": power, "buckets": [], "n": 0}
    found = tables.g_proposed & (tables.g_score >= std_threshold)
    missed = (~found).astype(float)
    unproposed = (~tables.g_proposed).astype(float)
    u = 1.0 - np.where(tables.g_proposed, tables.g_score, 0.0)

    b = bucket_indices(occ, bucket_edges)
    buckets = []
    for k in range(len(bucket_edges) - 1):
        m = b == k
        nk = int(m.sum())
        row = {"bucket": k, "lo": bucket_edges[k], "hi": bucket_edges[k + 1], "n": nk,
               "n_images": int(len(np.unique(img[m]))), "reliable": nk >= min_bucket_n}
        if nk:
            row.update({
                "miss_rate": float(missed[m].mean()),
                "miss_rate_ci": bootstrap_mean_ci(missed[m], img[m], boot_w),
                "unproposed_rate": float(unproposed[m].mean()),
                "mean_uncertainty": float(u[m].mean()),
                "mean_uncertainty_ci": bootstrap_mean_ci(u[m], img[m], boot_w),
                "mean_context": float(tables.g_ctx[m].mean()),
            })
        buckets.append(row)
    reliable = [r for r in buckets if r["reliable"]]
    rates = [r["miss_rate"] for r in reliable]
    monotone = bool(len(rates) >= 2 and all(a <= c + 1e-12 for a, c in zip(rates, rates[1:])))

    covars = np.column_stack([tables.g_area, _class_dummies(tables.g_cls)])
    rho = spearman(occ, u)
    rho_partial = partial_spearman(occ, u, covars)
    rho_ctx = spearman(occ, tables.g_ctx)
    rng = np.random.default_rng(seed)
    reps = {"rho": [], "partial": [], "ctx": []}
    for _ in range(n_rank_boot):
        rows = resample_rows_by_image(img, tables.n_images, rng)
        if len(rows) < 10:
            continue
        reps["rho"].append(spearman(occ[rows], u[rows]))
        reps["partial"].append(partial_spearman(occ[rows], u[rows], covars[rows]))
        reps["ctx"].append(spearman(occ[rows], tables.g_ctx[rows]))

    # logistic regression: missed ~ occlusion (per 0.1) + standardised log size + class
    area_z = (tables.g_area - tables.g_area.mean()) / max(tables.g_area.std(), 1e-9)
    X = np.column_stack([np.ones(n), occ * 10.0, area_z, _class_dummies(tables.g_cls)])
    logit = {"fitted": False}
    if 0 < missed.sum() < n and np.ptp(occ) > 0:
        beta = logistic_irls(X, missed)
        betas = []
        for bw in boot_w[:n_logit_boot]:
            w = bw[img]
            if (w * missed).sum() < 3 or (w * (1 - missed)).sum() < 3:
                continue
            betas.append(logistic_irls(X, missed, weights=w)[1])
        lo, hi = percentile_ci(betas)
        logit = {
            "fitted": True,
            "log_odds_per_0.1_occlusion": float(beta[1]),
            "odds_ratio_per_0.1_occlusion": float(np.exp(beta[1])),
            "odds_ratio_ci": (float(np.exp(lo)), float(np.exp(hi))) if np.isfinite(lo) else (None, None),
            "log_odds_per_sd_log_size": float(beta[2]),
            "controls": "log relative box area, class",
        }

    return {
        "n": int(n), "n_images": int(tables.n_images), "standard_threshold": std_threshold,
        "power": power, "buckets": buckets,
        **size_tertile_table(tables, std_threshold, bucket_edges, boot_w, min_bucket_n),
        "miss_rate_monotone_over_reliable_buckets": monotone,
        "spearman_occlusion_uncertainty": rho,
        "spearman_ci": percentile_ci(reps["rho"]),
        "partial_spearman_given_size_and_class": rho_partial,
        "partial_spearman_ci": percentile_ci(reps["partial"]),
        "spearman_occlusion_vs_deployable_context": rho_ctx,
        "spearman_context_ci": percentile_ci(reps["ctx"]),
        "logistic_miss_model": logit,
    }
