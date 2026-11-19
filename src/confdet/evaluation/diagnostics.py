"""Phase 1b (v0.6) analyses on cached predictions.   Used by scripts/17_phase1b.py.

  paired_recovery        RQ2 recovery of one score against ANOTHER score (not only against
                         confidence), with paired image-bootstrap intervals.
  quantile_strata_edges  occlusion-context strata with equal calib object counts (granularity
                         ablation). Rule fixed before test was looked at: 4 strata at the
                         calib quantiles of o_hat; tied edges collapse; any stratum with fewer
                         than `min_count` calib objects is merged into its smaller neighbour.
  occluder_diagnostic    for heavily occluded vehicles (annotated o_j >= 0.5) whose deployable
                         occlusion context is low (o_hat < 0.30): what happened to the vehicle
                         hiding them (detected / detected but treated as a duplicate / below the
                         context-confidence cutoff / never proposed).
"""
from __future__ import annotations

import numpy as np

from confdet.data.occlusion import intersection_matrix, iou_matrix
from confdet.evaluation.rq2_recovery import BASELINE, recovery_analysis


def paired_recovery(tables, scores: dict[str, np.ndarray], reference: str, std_threshold: float, budgets,
                    boot_w: np.ndarray, occluded_threshold: float = 0.2) -> dict:
    """Recovery of every score in `scores` compared with `scores[reference]` (paired bootstrap,
    same resamples). Internally the reference takes the place of the baseline."""
    if reference not in scores:
        raise KeyError(reference)
    renamed = {BASELINE: scores[reference], **{k: v for k, v in scores.items() if k != reference}}
    r = recovery_analysis(tables, renamed, std_threshold, budgets, boot_w, occluded_threshold)
    out = {"reference": reference, "comparisons": {}}
    for k, e in r["policies"].items():
        if k == BASELINE:
            continue
        out["comparisons"][f"{k} minus {reference}"] = {
            "all_misses": e.get("mean_recovery_delta_vs_confidence"),
            "all_misses_ci": e.get("mean_recovery_delta_ci"),
            "prob_better": e.get("prob_better_than_confidence"),
            "occluded_misses": e.get("mean_recovery_occluded_delta_vs_confidence"),
            "occluded_misses_ci": e.get("mean_recovery_occluded_delta_ci"),
            "prob_better_occluded": e.get("prob_better_occluded")}
    return out


def quantile_strata_edges(ctx: np.ndarray, n_strata: int = 4, min_count: int = 300) -> dict:
    """Edges over [0, 1] at the calib quantiles of o_hat, then merging (see module docstring)."""
    ctx = np.clip(np.asarray(ctx, float), 0.0, 1.0)
    qs = np.quantile(ctx, np.linspace(0, 1, n_strata + 1)[1:-1]) if len(ctx) else np.zeros(0)
    edges = sorted({0.0, 1.0, *[float(q) for q in qs if 0.0 < q < 1.0]})

    def counts(e):
        idx = np.clip(np.searchsorted(e, ctx, side="right") - 1, 0, len(e) - 2)
        return np.bincount(idx, minlength=len(e) - 1)

    merges = []
    c = counts(edges)
    while len(edges) > 2 and c.min() < min_count:
        i = int(np.argmin(c))
        # merge stratum i with its smaller neighbour (drop the edge between them)
        if i == 0:
            drop = 1
        elif i == len(c) - 1:
            drop = len(edges) - 2
        else:
            drop = i if c[i - 1] <= c[i + 1] else i + 1
        merges.append({"stratum": i, "count": int(c[i]), "dropped_edge": edges[drop]})
        edges = edges[:drop] + edges[drop + 1:]
        c = counts(edges)
    return {"edges": edges, "calib_counts": c.tolist(), "requested_strata": n_strata, "min_count": min_count,
            "quantiles": [float(q) for q in qs], "merges": merges}


def occluder_diagnostic(records, tables, context_conf: float, duplicate_iou: float = 0.5,
                        min_occlusion: float = 0.5, max_context: float = 0.30) -> dict:
    """Class-agnostic tables of a dataset without class maps (rows follow the records)."""
    cats = ("occluder_detected", "occluder_detected_but_duplicate", "occluder_below_cutoff",
            "occluder_not_proposed", "no_nearer_overlapping_vehicle")
    counts = {k: 0 for k in cats}
    by_target = {"target_proposed": {k: 0 for k in cats}, "target_unproposed": {k: 0 for k in cats}}
    n_heavy = n_low = 0
    g_off = c_off = 0
    for r in records:
        g = np.asarray(r.gt_xyxy, float).reshape(-1, 4)
        c = np.asarray(r.cand_xyxy, float).reshape(-1, 4)
        ng, nc = len(g), len(c)
        if ng:
            rows = np.arange(g_off, g_off + ng)
            heavy = tables.g_occ[rows] >= min_occlusion
            n_heavy += int(heavy.sum())
            low = heavy & (tables.g_ctx[rows] < max_context)
            n_low += int(low.sum())
            if low.any():
                inter = intersection_matrix(g, g)
                nearer = g[:, 3][None, :] > g[:, 3][:, None]           # bottom_y proxy: column nearer than row
                np.fill_diagonal(inter, 0.0)
                for j in np.nonzero(low)[0]:
                    cand = np.where(nearer[j], inter[j], 0.0)
                    key = "no_nearer_overlapping_vehicle"
                    if cand.max() > 0:
                        k = int(np.argmax(cand))
                        gk = g_off + k
                        if not tables.g_proposed[gk]:
                            key = "occluder_not_proposed"
                        elif tables.g_score[gk] < context_conf:
                            key = "occluder_below_cutoff"
                        else:
                            ci = int(tables.g_cand[gk]) - c_off
                            # the target's own box (its matched candidate, or the annotation if unproposed)
                            tb = c[int(tables.g_cand[g_off + j]) - c_off] if tables.g_proposed[g_off + j] else g[j]
                            key = ("occluder_detected_but_duplicate"
                                   if iou_matrix(c[ci][None], tb[None])[0, 0] >= duplicate_iou else "occluder_detected")
                    counts[key] += 1
                    by_target["target_proposed" if tables.g_proposed[g_off + j] else "target_unproposed"][key] += 1
        g_off += ng
        c_off += nc
    total = max(n_low, 1)
    return {"min_occlusion": min_occlusion, "max_context": max_context, "context_conf": context_conf,
            "n_heavily_occluded": n_heavy, "n_with_low_context": n_low,
            "share_with_low_context": n_low / n_heavy if n_heavy else None,
            "counts": counts, "shares": {k: v / total for k, v in counts.items()}, "by_target": by_target}


# ---------------------------------------------------------------------------
# Evidential pilot: is the reversed epistemic-rarity association a fine-class effect? (calib only)
# ---------------------------------------------------------------------------
def rarity_diagnostic(epistemic, aleatoric, fine_cls, has_box, fine_counts, fine_to_coarse, causes, log_area, img,
                      n_images, names, car_subtypes=("Hatchback", "Sedan", "SUV"), rare_share=0.02,
                      n_boot=300, seed=0) -> dict:
    """Per vehicle (only those with a box: `has_box`): epistemic / aleatoric uncertainty of its box,
    fine class, RQ5 miss cause (rq5_miss_causes.CAUSES index, -1 = found).

    (a) Spearman / partial (log size) of epistemic vs FINE rarity and vs COARSE rarity, where the coarse
        rarity of a vehicle is -log of the fit-role share of its common class (fine_to_coarse[fine], -1 = none)
    (b) the same within confusion misses and within visibility misses (suppressed / merged / unproposed)
    (c) mean epistemic and aleatoric per fine class, and car subtypes vs rare classes (share < rare_share),
        with an image-bootstrap interval for the difference of means
    """
    from confdet.evaluation.rq5_miss_causes import CAUSES, VISIBILITY
    from confdet.evaluation.stats import partial_spearman, percentile_ci, resample_rows_by_image, spearman
    fine_counts = np.asarray(fine_counts, float)
    share = fine_counts / max(fine_counts.sum(), 1)
    f2c = np.asarray(fine_to_coarse)
    n_coarse = int(f2c.max()) + 1 if (f2c >= 0).any() else 0
    coarse_counts = np.bincount(f2c[f2c >= 0], weights=fine_counts[f2c >= 0], minlength=n_coarse)
    coarse_share = coarse_counts / max(coarse_counts.sum(), 1)
    m = np.asarray(has_box, bool) & (f2c[np.asarray(fine_cls)] >= 0)
    epi, alea = np.asarray(epistemic, float)[m], np.asarray(aleatoric, float)[m]
    cls = np.asarray(fine_cls)[m]
    fine_r = -np.log(np.clip(share[cls], 1e-12, None))
    coarse_r = -np.log(np.clip(coarse_share[f2c[cls]], 1e-12, None))
    size = np.asarray(log_area, float)[m][:, None]
    im = np.asarray(img)[m]
    cause = np.asarray(causes)[m]
    rng = np.random.default_rng(seed)

    def assoc(sel, u):
        if sel.sum() < 30:
            return {"n": int(sel.sum()), "note": "fewer than 30 vehicles"}
        out = {"n": int(sel.sum())}
        for nm, r in (("fine_rarity", fine_r), ("coarse_rarity", coarse_r)):
            reps = []
            for _ in range(n_boot):
                rows = resample_rows_by_image(im[sel], n_images, rng)
                if len(rows) > 10:
                    reps.append(spearman(u[sel][rows], r[sel][rows]))
            out[nm] = {"spearman": spearman(u[sel], r[sel]), "spearman_ci": percentile_ci(reps),
                       "partial_spearman_size": partial_spearman(u[sel], r[sel], size[sel])}
        return out

    all_ = np.ones(len(epi), bool)
    confused = cause == CAUSES.index("confused")
    visibility = np.isin(cause, [CAUSES.index(c) for c in VISIBILITY])
    per_class = []
    for c in np.unique(cls):
        s = cls == c
        per_class.append({"class": names[c], "fit_share": float(share[c]), "n": int(s.sum()),
                          "mean_epistemic": float(epi[s].mean()), "mean_aleatoric": float(alea[s].mean())})
    per_class.sort(key=lambda r: -r["fit_share"])
    sub_ids = [names.index(n) for n in car_subtypes if n in names]
    rare_ids = [i for i in range(len(names)) if share[i] < rare_share and f2c[i] >= 0]
    sub, rare = np.isin(cls, sub_ids), np.isin(cls, rare_ids)
    diffs = []
    for _ in range(n_boot):
        rows = resample_rows_by_image(im, n_images, rng)
        a, b = epi[rows][sub[rows]], epi[rows][rare[rows]]
        if len(a) and len(b):
            diffs.append(a.mean() - b.mean())
    return {"n_vehicles_with_box": int(m.sum()),
            "a_epistemic_all": assoc(all_, epi), "a_aleatoric_all": assoc(all_, alea),
            "b_epistemic_confusion_misses": assoc(confused, epi), "b_epistemic_visibility_misses": assoc(visibility, epi),
            "c_per_class": per_class,
            "c_car_subtypes_vs_rare": {"car_subtypes": [names[i] for i in sub_ids], "rare_classes": [names[i] for i in rare_ids],
                                       "rare_share_below": rare_share,
                                       "mean_epistemic_car_subtypes": float(epi[sub].mean()) if sub.any() else None,
                                       "mean_epistemic_rare": float(epi[rare].mean()) if rare.any() else None,
                                       "difference": float(epi[sub].mean() - epi[rare].mean()) if sub.any() and rare.any() else None,
                                       "difference_ci": percentile_ci(diffs)}}
