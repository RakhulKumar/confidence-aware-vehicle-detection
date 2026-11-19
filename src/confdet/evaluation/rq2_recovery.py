"""RQ2 -- can the post-hoc layer recover occlusion-driven misses at an
acceptable false-alarm cost?   docs/theoretical_foundations.md section 7.2.

Setting. The detector's normal output (confidence >= standard threshold) is
left untouched. Below it lies a pool of weak candidates; some of them are
real vehicles the standard threshold silently dropped. A POLICY is a score
on those weak candidates; raising more of them into a "flagged: possible
vehicle" tier recovers misses and costs false alarms.

Policies compared, all on the same labelled candidates:

  confidence        lower the threshold everywhere (the naive baseline; any
                    monotone recalibration of confidence gives this curve)
  mondrian_rank     occlusion-context-conditional conformal rank
                    (uq/conformal.rank_transform)
  occlusion_aware   fitted P(true positive | confidence, occlusion context)
                    (uq/calibrator)

The comparison is made at MATCHED COST: for a budget of b extra false alarms
per image, what fraction of the standard threshold's misses does each policy
recover -- over all misses and over occluded misses? Differences from the
baseline come with paired image-bootstrap intervals.

Two things the previous version got wrong and this one does not:
  * it decided where to relax using the ground-truth occlusion of the very
    object being recovered (information no deployed system has), while
    charging false alarms everywhere;
  * it compared nothing against anything: no baseline at equal cost.

A ceiling is reported alongside: misses with no candidate at all cannot be
recovered by ANY post-hoc rule.
"""
from __future__ import annotations

import numpy as np

from confdet.data.occlusion import bucket_indices
from confdet.evaluation.stats import bootstrap_mean_ci, percentile_ci
from confdet.uq.conformal import ConformalThresholds, group_indices, keep_mask, rank_transform

BASELINE = "confidence"


def policy_scores(tables, mondrian: ConformalThresholds | None, calibrator) -> dict[str, np.ndarray]:
    """Score of every candidate under every available policy."""
    out = {BASELINE: tables.c_conf.astype(float)}
    if mondrian is not None:
        out["mondrian_rank"] = rank_transform(tables.c_conf, tables.c_ctx, mondrian)
    if calibrator is not None:
        out["occlusion_aware"] = calibrator.predict(tables.c_conf, tables.c_ctx, tables.c_dup)
    return out


class _Curve:
    """Recovery-vs-extra-false-alarms curve of one policy over the weak
    candidates, re-evaluable under any image weighting.
    """

    def __init__(self, score, img, tp, tp_occluded):
        order = np.argsort(-score, kind="stable")
        s = score[order]
        self.img, self.tp, self.occ = img[order], tp[order].astype(float), tp_occluded[order].astype(float)
        # candidates with equal score enter together: evaluate only where the score changes
        self.last = np.nonzero(np.r_[s[1:] != s[:-1], True])[0] if len(s) else np.zeros(0, dtype=int)

    def at_budgets(self, w_img, budgets, n_missed, n_missed_occ):
        """(recovery_all, recovery_occluded) arrays, one entry per budget."""
        n_b = len(budgets)
        if len(self.last) == 0 or w_img.sum() <= 0:
            return np.zeros(n_b), np.zeros(n_b)
        w = w_img[self.img]
        fp = np.cumsum(w * (1 - self.tp))[self.last] / w_img.sum()
        tp = np.cumsum(w * self.tp)[self.last]
        oc = np.cumsum(w * self.occ)[self.last]
        k = np.searchsorted(fp, np.asarray(budgets, dtype=float), side="right") - 1
        ok = k >= 0
        kk = np.clip(k, 0, None)
        rec = np.where(ok, tp[kk], 0.0) / n_missed if n_missed > 0 else np.zeros(n_b)
        rec_o = np.where(ok, oc[kk], 0.0) / n_missed_occ if n_missed_occ > 0 else np.zeros(n_b)
        return rec, rec_o

    def points(self, n_images, n_missed, n_missed_occ, max_fp, max_points=150):
        if len(self.last) == 0:
            return []
        fp = np.cumsum(1 - self.tp)[self.last] / max(n_images, 1)
        tp = np.cumsum(self.tp)[self.last] / max(n_missed, 1)
        oc = np.cumsum(self.occ)[self.last] / max(n_missed_occ, 1)
        m = fp <= max_fp
        fp, tp, oc = fp[m], tp[m], oc[m]
        if len(fp) > max_points:
            sel = np.unique(np.linspace(0, len(fp) - 1, max_points).astype(int))
            fp, tp, oc = fp[sel], tp[sel], oc[sel]
        return [{"extra_fp_per_image": float(a), "recovery": float(b), "recovery_occluded": float(c)}
                for a, b, c in zip(fp, tp, oc)]


def recovery_analysis(
    tables,
    scores: dict[str, np.ndarray],
    std_threshold: float,
    budgets: list[float],
    boot_w: np.ndarray,
    occluded_threshold: float = 0.2,
) -> dict:
    found = tables.g_proposed & (tables.g_score >= std_threshold)
    missed = ~found
    missed_occ = missed & (tables.g_occ >= occluded_threshold)
    n_missed, n_missed_occ = int(missed.sum()), int(missed_occ.sum())
    n_images = tables.n_images

    weak = tables.c_conf < std_threshold
    idx = np.nonzero(weak)[0]
    tp = tables.c_tp[idx]
    if tables.n_gt:
        tp_occ = tp & (tables.g_occ[np.clip(tables.c_gt[idx], 0, None)] >= occluded_threshold)
    else:
        tp_occ = np.zeros(len(idx), dtype=bool)
    img = tables.c_img[idx]

    out = {
        "standard_threshold": std_threshold,
        "n_images": n_images,
        "n_missed_at_standard": n_missed,
        "n_missed_occluded": n_missed_occ,
        "occluded_threshold": occluded_threshold,
        "n_weak_candidates": int(len(idx)),
        # no post-hoc rule can do better than this:
        "recoverable_ceiling": float((missed & tables.g_proposed).sum() / n_missed) if n_missed else None,
        "recoverable_ceiling_occluded":
            float((missed_occ & tables.g_proposed).sum() / n_missed_occ) if n_missed_occ else None,
        "budgets_extra_fp_per_image": [float(b) for b in budgets],
        "policies": {},
    }
    if n_missed == 0 or len(idx) == 0:
        out["note"] = "nothing to recover: no misses at the standard threshold, or no weak candidates"
        return out

    budgets = [float(b) for b in budgets]
    grid = np.linspace(0, max(budgets), 41)[1:]
    ones = np.ones(n_images)
    w_missed = np.bincount(tables.g_img, weights=missed.astype(float), minlength=n_images)
    w_missed_occ = np.bincount(tables.g_img, weights=missed_occ.astype(float), minlength=n_images)

    curves = {name: _Curve(np.asarray(s, dtype=float)[idx], img, tp, tp_occ) for name, s in scores.items()}
    reps: dict[str, dict[str, list]] = {n: {"all": [], "occ": [], "mean_all": [], "mean_occ": []} for n in curves}
    for bw in boot_w:
        nm, nmo = float(bw @ w_missed), float(bw @ w_missed_occ)
        for name, cv in curves.items():
            a, o = cv.at_budgets(bw, budgets, nm, nmo)
            ga, go = cv.at_budgets(bw, grid, nm, nmo)
            reps[name]["all"].append(a); reps[name]["occ"].append(o)
            reps[name]["mean_all"].append(ga.mean()); reps[name]["mean_occ"].append(go.mean())
    reps = {n: {k: np.array(v) for k, v in d.items()} for n, d in reps.items()}

    for name, cv in curves.items():
        a, o = cv.at_budgets(ones, budgets, n_missed, n_missed_occ)
        ga, go = cv.at_budgets(ones, grid, n_missed, n_missed_occ)
        rows = []
        for i, b in enumerate(budgets):
            row = {"extra_fp_per_image": b,
                   "recovery": float(a[i]), "recovery_ci": percentile_ci(reps[name]["all"][:, i]),
                   "recovery_occluded": float(o[i]) if n_missed_occ else None,
                   "recovery_occluded_ci": percentile_ci(reps[name]["occ"][:, i]) if n_missed_occ else None}
            if name != BASELINE and BASELINE in curves:
                d_all = reps[name]["all"][:, i] - reps[BASELINE]["all"][:, i]
                d_occ = reps[name]["occ"][:, i] - reps[BASELINE]["occ"][:, i]
                row["delta_vs_confidence_ci"] = percentile_ci(d_all)
                row["prob_better_than_confidence"] = float(np.mean(d_all > 0))
                row["delta_occluded_vs_confidence_ci"] = percentile_ci(d_occ) if n_missed_occ else None
                row["prob_better_occluded"] = float(np.mean(d_occ > 0)) if n_missed_occ else None
            rows.append(row)
        entry = {
            "at_budgets": rows,
            "mean_recovery_over_budget_range": float(ga.mean()),
            "mean_recovery_ci": percentile_ci(reps[name]["mean_all"]),
            "mean_recovery_occluded": float(go.mean()) if n_missed_occ else None,
            "mean_recovery_occluded_ci": percentile_ci(reps[name]["mean_occ"]) if n_missed_occ else None,
            "curve": cv.points(n_images, n_missed, n_missed_occ, max(budgets)),
        }
        out["policies"][name] = entry

    # point-estimate deltas (kept separate from the bootstrap so they are exact)
    if BASELINE in curves:
        base_a, base_o = curves[BASELINE].at_budgets(ones, budgets, n_missed, n_missed_occ)
        base_ga, base_go = curves[BASELINE].at_budgets(ones, grid, n_missed, n_missed_occ)
        for name, cv in curves.items():
            if name == BASELINE:
                continue
            a, o = cv.at_budgets(ones, budgets, n_missed, n_missed_occ)
            ga, go = cv.at_budgets(ones, grid, n_missed, n_missed_occ)
            for i, row in enumerate(out["policies"][name]["at_budgets"]):
                row["delta_vs_confidence"] = float(a[i] - base_a[i])
                row["delta_occluded_vs_confidence"] = float(o[i] - base_o[i]) if n_missed_occ else None
            d = reps[name]["mean_all"] - reps[BASELINE]["mean_all"]
            do = reps[name]["mean_occ"] - reps[BASELINE]["mean_occ"]
            out["policies"][name]["mean_recovery_delta_vs_confidence"] = float(ga.mean() - base_ga.mean())
            out["policies"][name]["mean_recovery_delta_ci"] = percentile_ci(d)
            out["policies"][name]["prob_better_than_confidence"] = float(np.mean(d > 0))
            if n_missed_occ:
                out["policies"][name]["mean_recovery_occluded_delta_vs_confidence"] = float(go.mean() - base_go.mean())
                out["policies"][name]["mean_recovery_occluded_delta_ci"] = percentile_ci(do)
                out["policies"][name]["prob_better_occluded"] = float(np.mean(do > 0))
    return out


def conformal_report(
    tables,
    rules: dict[str, tuple[ConformalThresholds, np.ndarray]],
    std_threshold: float,
    bucket_edges: list[float],
    boot_w: np.ndarray,
    min_bucket_n: int = 30,
) -> dict:
    """Does each conformal rule deliver its promised recall on held-out
    objects -- overall, inside each occlusion-context group (where the
    Mondrian guarantee lives), and inside each ANNOTATED occlusion bucket
    (not guaranteed; this is the honest check of whether conditioning on the
    deployable context fixes the occluded strata)?

    rules: name -> (thresholds, per-candidate score array the thresholds apply to).
    The row "standard_threshold" is the detector's unmodified output.
    """
    img = tables.g_img
    n_images = tables.n_images
    b_occ = bucket_indices(tables.g_occ, bucket_edges)
    accepted = tables.c_conf >= std_threshold
    found_std = tables.g_proposed & (tables.g_score >= std_threshold)

    def by_bucket(found):
        rows = []
        for k in range(len(bucket_edges) - 1):
            m = b_occ == k
            nk = int(m.sum())
            rows.append({"bucket": k, "lo": bucket_edges[k], "hi": bucket_edges[k + 1], "n": nk,
                         "reliable": nk >= min_bucket_n,
                         "coverage": float(found[m].mean()) if nk else None,
                         "coverage_ci": bootstrap_mean_ci(found[m].astype(float), img[m], boot_w) if nk else None})
        return rows

    out = {"standard_threshold": {
        "coverage": float(found_std.mean()) if tables.n_gt else None,
        "coverage_ci": bootstrap_mean_ci(found_std.astype(float), img, boot_w),
        "by_occlusion_bucket": by_bucket(found_std),
        "fp_per_image": float((accepted & ~tables.c_tp).sum() / max(n_images, 1)),
    }}
    for name, (thr, cand_score) in rules.items():
        cand_score = np.asarray(cand_score, dtype=float)
        ctx_c = tables.c_ctx if len(thr.context_edges) > 2 else None
        ctx_g = tables.g_ctx if len(thr.context_edges) > 2 else None
        kept = keep_mask(cand_score, ctx_c, thr)
        obj_score = np.where(tables.g_proposed, cand_score[np.clip(tables.g_cand, 0, None)]
                             if tables.n_cand else 0.0, 0.0)
        g_group = group_indices(ctx_g, tables.n_gt, thr)
        t = np.array([g.threshold for g in thr.groups])[g_group] if tables.n_gt else np.zeros(0)
        found = tables.g_proposed & (obj_score >= t)
        flagged = kept & ~accepted
        groups = []
        for gi, g in enumerate(thr.groups):
            m = g_group == gi
            strata = [s for s, gg in enumerate(thr.group_of_stratum) if gg == gi]
            groups.append({
                "group": gi,
                "context_range": [thr.context_edges[min(strata)], thr.context_edges[max(strata) + 1]],
                "n_calibration": g.n, "n_test": int(m.sum()),
                "threshold": None if not np.isfinite(g.threshold) else float(g.threshold),
                "achievable": bool(g.achievable), "min_certifiable_alpha": float(g.min_alpha),
                "reason": g.reason,
                "coverage": float(found[m].mean()) if m.any() else None,
                "coverage_ci": bootstrap_mean_ci(found[m].astype(float), img[m], boot_w) if m.any() else None,
            })
        system_found = found | found_std
        out[name] = {
            "alpha": thr.alpha, "target_coverage": 1 - thr.alpha, "score": thr.score_name,
            "conditional": thr.is_conditional,
            "all_groups_achievable": bool(all(g.achievable for g in thr.groups)),
            "coverage": float(found.mean()) if tables.n_gt else None,
            "coverage_ci": bootstrap_mean_ci(found.astype(float), img, boot_w),
            "by_context_group": groups,
            # the rule on its own -- this is what the conformal guarantee is about
            "by_occlusion_bucket": by_bucket(found),
            # the deployed system = normal output + flagged tier. When the rule's
            # threshold is stricter than the standard one (the detector already
            # exceeds the target), the rule flags nothing and system == standard.
            "system_recall": float(system_found.mean()) if tables.n_gt else None,
            "system_recall_ci": bootstrap_mean_ci(system_found.astype(float), img, boot_w),
            "system_by_occlusion_bucket": by_bucket(system_found),
            "adds_flags": bool(flagged.any()),
            "flagged_per_image": float(flagged.sum() / max(n_images, 1)),
            "flagged_false_per_image": float((flagged & ~tables.c_tp).sum() / max(n_images, 1)),
            "flagged_true_total": int((flagged & tables.c_tp).sum()),
        }
    return out
