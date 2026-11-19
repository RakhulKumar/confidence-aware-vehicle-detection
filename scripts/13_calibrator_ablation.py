#!/usr/bin/env python3
"""Ablation of the occlusion-aware candidate score (see evaluation/ablation.py).

  python scripts/13_calibrator_ablation.py [--detectors yolo26 yolov8 rtdetr] [--matching class_agnostic]
                                           [--folds 5]

For each detector, in this order:

  1. Load the CALIB predictions only. Choose a feature set by 5-fold
     cross-validation over calib images (criterion fixed in advance: mean
     recovery of all misses over the extra-false-alarm budget range on the
     held-out folds). Write
     outputs/<experiment>/<detector>/<mode>/ablation_selection.json.
  2. Only then load the TEST predictions. Fit every set as fit_posthoc fits
     the score (same calib rows, same seed) and evaluate each once against
     plain confidence with paired image-bootstrap intervals. Write
     .../ablation_test.json.

No GPU. Changes nothing in the configured post-hoc layer: the default
calibrator stays what configs/experiment.yaml and scripts/06 say.
"""
from __future__ import annotations

from datetime import datetime, timezone

from _common import setup

from confdet.evaluation.ablation import ABLATION_SETS, cv_select, evaluate_on_test
from confdet.evaluation.records import load_records
from confdet.evaluation.stats import bootstrap_image_weights
from confdet.pipeline import PostHoc, calibrator_candidate_rows, tables_from_records
from confdet.utils.io_utils import load_results, save_results
from confdet.utils.paths import calibration_path, predictions_path


def _ci(c):
    return "[n/a]" if not c or c[0] is None or c[0] != c[0] else f"[{c[0]:+.4f}, {c[1]:+.4f}]"


def run(det: str, mode: str, n_folds: int, cfg: dict, log) -> None:
    ev, ph = cfg["evaluation"], cfg["posthoc"]
    dev = cfg["experiment"]["development"]
    seed = int(cfg["project"].get("seed", 42))
    occ_thr = float(cfg["occlusion"].get("occluded_threshold", 0.2))
    dup_iou = float(ph.get("duplicate_iou", 0.5))
    budgets = ev["extra_fp_per_image_budgets"]
    cal_file = calibration_path(cfg, det, mode)
    if not cal_file.exists():
        log.warning(f"[{mode}] {det}: no calibration.json; run scripts/06 first. Skipped.")
        return
    posthoc = PostHoc(load_results(cal_file))
    tau, ctx = posthoc.tau, posthoc.context_conf
    out_dir = cal_file.parent

    # ---- 1. selection on calib only -------------------------------------------------------------
    calib = tables_from_records(cfg, load_records(predictions_path(cfg, det, dev, "calib")), mode, context_conf=ctx)
    log.info(f"===== {det} | {mode} | tau {tau:.2f} | calib: {calib.n_images} images, {calib.n_cand} candidates =====")
    sel = cv_select(calib, tau, budgets, occ_thr, n_folds, seed, dup_iou)
    sel.update({"detector": det, "matching": mode, "written_at_utc": datetime.now(timezone.utc).isoformat(),
                "note": "computed from calib predictions only; test predictions had not been loaded"})
    save_results(sel, out_dir / "ablation_selection.json")
    for name in sel["ranking"]:
        r = sel["table"][name]
        log.info(f"  CV {name:<12} mean recovery {r['cv_mean_recovery']:.4f} (sd over folds "
                 f"{r['cv_mean_recovery_sd_over_folds']:.4f}) | occluded {r['cv_mean_recovery_occluded']:.4f}")
    log.info(f"  chosen on calib: {sel['chosen']}  -> saved {out_dir / 'ablation_selection.json'}")

    # ---- 2. test, once ----------------------------------------------------------------------------
    test = tables_from_records(cfg, load_records(predictions_path(cfg, det, dev, "test")), mode, context_conf=ctx)
    bw = bootstrap_image_weights(test.n_images, int(ev.get("bootstrap_replicates", 500)), seed)
    res = evaluate_on_test(calib, test, tau, budgets, bw, calibrator_candidate_rows(calib, cfg, seed),
                           occ_thr, dup_iou)
    res.update({"detector": det, "matching": mode, "std_threshold": tau, "chosen_on_calib": sel["chosen"],
                "feature_sets": {k: v for k, v in ABLATION_SETS.items()},
                "note": "every set evaluated once on test; the choice was fixed on calib beforehand "
                        "(ablation_selection.json). The configured default calibrator is unchanged."})
    save_results(res, out_dir / "ablation_test.json")
    for name, r in res["sets"].items():
        if "error" in r:
            log.warning(f"  test {name:<12} not fitted: {r['error']}")
            continue
        log.info(f"  test {name:<12} mean recovery {r['mean_recovery_over_budget_range']:.4f} | vs confidence "
                 f"{r['mean_recovery_delta_vs_confidence']:+.4f} {_ci(r['mean_recovery_delta_ci'])} | occluded "
                 f"{r['mean_recovery_occluded_delta_vs_confidence']:+.4f} {_ci(r['mean_recovery_occluded_delta_ci'])}"
                 + ("   <- chosen on calib" if name == sel["chosen"] else ""))
    log.info(f"  saved {out_dir / 'ablation_test.json'}")


def main():
    def extra(p):
        p.add_argument("--detectors", nargs="+", default=None)
        p.add_argument("--matching", default="class_agnostic")
        p.add_argument("--folds", type=int, default=5)

    args, cfg, log = setup("calibrator_ablation", __doc__, extra=extra)
    for det in args.detectors or list(cfg["detectors"]["models"]):
        run(det, args.matching, args.folds, cfg, log)


if __name__ == "__main__":
    main()
