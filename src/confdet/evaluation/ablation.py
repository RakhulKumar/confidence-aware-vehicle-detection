"""Ablation of the occlusion-aware candidate score: which features carry the
RQ2 gain?   Used by scripts/13_calibrator_ablation.py.

Feature sets (uq/calibrator.FEATURE_SETS):

  z_only       z                                  (Platt: ranks exactly like confidence)
  z_dup        z, dup
  z_ctx        z, ctx, z*ctx
  full         z, ctx, dup, z*ctx, z*dup          (the current default)
  full_strict  full with dup -> dupS = dup if dup >= duplicate_iou else 0,
               so the duplicate feature and occlusion context are built from
               disjoint sets of neighbouring boxes

Protocol, fixed before any test number is seen:

  1. SELECTION ON CALIB ONLY. 5-fold cross-validation over calib IMAGES.
     Each set is fitted on four folds' candidates and scored on the fifth.
     Criterion: mean recovery of ALL misses over the extra-false-alarm budget
     range (the RQ2 headline quantity, same grid), averaged over held-out
     folds. Occluded-miss recovery is reported alongside but does not decide.
  2. TEST, ONCE. Every set is fitted as pipeline.fit_posthoc fits the score
     (the calibrator half of calib) and evaluated on test with paired image-
     bootstrap intervals against plain confidence, overall and on occluded
     misses (rq2_recovery.recovery_analysis).
"""
from __future__ import annotations

import numpy as np

from confdet.evaluation.rq2_recovery import _Curve, recovery_analysis
from confdet.evaluation.simulate import stable_seed
from confdet.uq.calibrator import fit_calibrator

ABLATION_SETS = {
    "z_only": "confidence",
    "z_dup": "z_dup",
    "z_ctx": "z_ctx",
    "full": "occlusion_aware",
    "full_strict": "full_strict",
}
CRITERION = ("mean recovery of all misses over the extra-false-alarm budget range, held-out calib folds "
             "(5-fold CV by image), averaged over folds; higher is better")


def image_folds(image_ids: list[str], n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index per image, deterministic and independent of image order."""
    return np.array([stable_seed(i, seed + 1013) % n_folds for i in image_ids], dtype=int)


def _budget_grid(budgets) -> np.ndarray:
    """Same grid as rq2_recovery.recovery_analysis uses for its mean recovery."""
    return np.linspace(0, max(float(b) for b in budgets), 41)[1:]


def heldout_mean_recovery(tables, score: np.ndarray, std_threshold: float, budgets, occluded_threshold: float,
                          image_mask: np.ndarray) -> tuple[float, float]:
    """Mean recovery (all misses, occluded misses) over the budget range,
    counting only the images in `image_mask`: false alarms per image,
    recovered misses and the number of misses are all restricted to them.
    """
    found = tables.g_proposed & (tables.g_score >= std_threshold)
    missed = ~found
    missed_occ = missed & (tables.g_occ >= occluded_threshold)
    w_img = np.asarray(image_mask, dtype=float)
    n_missed = float((w_img[tables.g_img] * missed).sum())
    n_missed_occ = float((w_img[tables.g_img] * missed_occ).sum())
    weak = np.nonzero(tables.c_conf < std_threshold)[0]
    tp = tables.c_tp[weak]
    tp_occ = tp & (tables.g_occ[np.clip(tables.c_gt[weak], 0, None)] >= occluded_threshold) if tables.n_gt \
        else np.zeros(len(weak), dtype=bool)
    cv = _Curve(np.asarray(score, dtype=float)[weak], tables.c_img[weak], tp, tp_occ)
    rec, rec_o = cv.at_budgets(w_img, _budget_grid(budgets), n_missed, n_missed_occ)
    return float(rec.mean()), (float(rec_o.mean()) if n_missed_occ > 0 else float("nan"))


def _fit(tables, rows: np.ndarray, feature_key: str, duplicate_iou: float):
    return fit_calibrator(tables.c_conf[rows], tables.c_ctx[rows], tables.c_dup[rows], tables.c_tp[rows],
                          feature_key, duplicate_iou=duplicate_iou)


def cv_select(calib, std_threshold: float, budgets, occluded_threshold: float = 0.2, n_folds: int = 5,
              seed: int = 0, duplicate_iou: float = 0.5, sets: dict[str, str] | None = None) -> dict:
    """Choose a feature set by cross-validation on the CALIB tables only."""
    sets = sets or ABLATION_SETS
    fold = image_folds(calib.image_ids, n_folds, seed)
    c_fold = fold[calib.c_img]
    table = {}
    for name, key in sets.items():
        per_fold = []
        for k in range(n_folds):
            held = fold == k
            row = {"fold": k, "n_images": int(held.sum())}
            try:
                cal = _fit(calib, c_fold != k, key, duplicate_iou)
            except ValueError as e:
                row["error"] = str(e)
                per_fold.append(row)
                continue
            score = cal.predict(calib.c_conf, calib.c_ctx, calib.c_dup)
            row["mean_recovery"], row["mean_recovery_occluded"] = heldout_mean_recovery(
                calib, score, std_threshold, budgets, occluded_threshold, held)
            per_fold.append(row)
        ok = [r for r in per_fold if "mean_recovery" in r]
        vals = np.array([r["mean_recovery"] for r in ok])
        occ = np.array([r["mean_recovery_occluded"] for r in ok], dtype=float)
        table[name] = {
            "feature_set": key,
            "n_folds_fitted": len(ok),
            "cv_mean_recovery": float(vals.mean()) if len(ok) else None,
            "cv_mean_recovery_sd_over_folds": float(vals.std(ddof=1)) if len(ok) > 1 else None,
            "cv_mean_recovery_occluded": float(np.nanmean(occ)) if np.isfinite(occ).any() else None,
            "folds": per_fold,
        }
    ranked = sorted((n for n in table if table[n]["n_folds_fitted"] == n_folds),
                    key=lambda n: -table[n]["cv_mean_recovery"])
    return {"criterion": CRITERION, "n_folds": n_folds, "fold_seed": seed, "std_threshold": std_threshold,
            "budgets": [float(b) for b in budgets], "occluded_threshold": occluded_threshold,
            "duplicate_iou": duplicate_iou, "n_calib_images": calib.n_images,
            "chosen": ranked[0] if ranked else None, "ranking": ranked, "table": table}


def evaluate_on_test(calib, test, std_threshold: float, budgets, boot_w: np.ndarray, calibrator_rows: np.ndarray,
                     occluded_threshold: float = 0.2, duplicate_iou: float = 0.5,
                     sets: dict[str, str] | None = None) -> dict:
    """Fit every set on the calib candidates in `calibrator_rows` (the same
    rows pipeline.fit_posthoc fits its score on) and evaluate all of them,
    once, on test against plain confidence with paired bootstrap intervals.
    """
    sets = sets or ABLATION_SETS
    scores, fitted = {"confidence": test.c_conf.astype(float)}, {}
    for name, key in sets.items():
        try:
            cal = _fit(calib, calibrator_rows, key, duplicate_iou)
        except ValueError as e:
            fitted[name] = {"error": str(e)}
            continue
        fitted[name] = cal.to_dict()
        scores[name] = cal.predict(test.c_conf, test.c_ctx, test.c_dup)
    rec = recovery_analysis(test, scores, std_threshold, budgets, boot_w, occluded_threshold)
    rows = {}
    for name in sets:
        e = rec["policies"].get(name)
        if e is None:
            rows[name] = {"error": fitted[name].get("error", "not evaluated")}
            continue
        rows[name] = {k: e.get(k) for k in (
            "mean_recovery_over_budget_range", "mean_recovery_ci", "mean_recovery_occluded",
            "mean_recovery_occluded_ci", "mean_recovery_delta_vs_confidence", "mean_recovery_delta_ci",
            "prob_better_than_confidence", "mean_recovery_occluded_delta_vs_confidence",
            "mean_recovery_occluded_delta_ci", "prob_better_occluded")}
        rows[name]["at_budgets"] = e["at_budgets"]
        rows[name]["calibrator"] = fitted[name]
    base = rec["policies"].get("confidence", {})
    return {"n_missed": rec["n_missed_at_standard"], "n_missed_occluded": rec["n_missed_occluded"],
            "recoverable_ceiling": rec["recoverable_ceiling"],
            "confidence": {k: base.get(k) for k in ("mean_recovery_over_budget_range", "mean_recovery_ci",
                                                    "mean_recovery_occluded", "mean_recovery_occluded_ci")},
            "sets": rows}
