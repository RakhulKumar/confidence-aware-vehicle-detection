#!/usr/bin/env python3
"""Stage 3: fit the post-hoc layer on the development dataset's calib role,
once per matching mode (evaluation.matching_modes):

  * the detector's operating threshold tau ("max_f1": the F1-optimal
    threshold of this detector on calib; or the fixed number in the config);
  * the occlusion-aware candidate score (uq/calibrator.py), on one half of
    the calib images;
  * miss-aware conformal thresholds -- global, occlusion-conditional
    (Mondrian) and on the occlusion-aware score -- on the other half, for
    every alpha in posthoc.alpha_grid.

  python scripts/06_calibrate.py [--detector NAME] [--matching MODE]

Writes outputs/<experiment>/<detector>/<mode>/calibration.json. No GPU.
Targets that cannot be certified (too many vehicles with no candidate at
all) are reported with the smallest miss rate that CAN be certified.
"""
from __future__ import annotations

from _common import setup

from confdet.evaluation.records import load_records
from confdet.pipeline import calibration_tables, fit_posthoc
from confdet.utils.io_utils import matching_modes, save_results
from confdet.utils.paths import calibration_path, predictions_path


def main():
    args, cfg, log = setup("calibrate", __doc__, detector=True,
                           extra=lambda p: p.add_argument("--matching", default=None))
    dev = cfg["experiment"]["development"]
    rs = load_records(predictions_path(cfg, args.detector, dev, "calib"))
    for mode in ([args.matching] if args.matching else matching_modes(cfg)):
        calib, tau, ctx = calibration_tables(cfg, rs, mode)
        log.info(f"[{mode}] calibration role: {calib.n_images} images, {calib.n_gt} vehicles, "
                 f"{calib.n_cand} candidates | operating threshold tau = {tau:.2f} "
                 f"({cfg['evaluation']['standard_operating_threshold']})")
        fitted = fit_posthoc(cfg, calib, seed=int(cfg["project"].get("seed", 42)), tau=tau, mode=mode)
        fitted["meta"].update({"detector": args.detector, "dataset": dev})
        for note in fitted["notes"]:
            log.warning(note)
        if fitted["calibrator"]:
            c = fitted["calibrator"]
            log.info("  occlusion-aware score: " + ", ".join(
                f"{n}={v:+.2f}" for n, v in zip(["intercept"] + c["features"], c["coef"])) + " (standardised)")
        main_alpha = f"{float(cfg['posthoc']['alpha']):g}"
        for rule, thr in fitted["thresholds"][main_alpha].items():
            for g in thr["groups"]:
                t = "keep all" if g["threshold"] is None else f"{g['threshold']:.4f}"
                status = "ok" if g["achievable"] else f"NOT CERTIFIABLE (min alpha {g['min_alpha']:.3f})"
                log.info(f"  alpha={main_alpha} {rule:<24} n={g['n']:<6} unproposed={g['n_unproposed']:<5} "
                         f"threshold={t:<9} {status}")
        out = calibration_path(cfg, args.detector, mode)
        save_results(fitted, out)
        log.info(f"  saved {out}")
    log.info(f"Next: python scripts/07_run_evaluation.py --detector {args.detector}")


if __name__ == "__main__":
    main()
