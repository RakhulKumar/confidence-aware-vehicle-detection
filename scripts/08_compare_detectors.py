#!/usr/bin/env python3
"""RQ3, second half: rank the detectors by reliability under each metric and
test whether a miss-aware metric orders them differently from D-ECE.

  python scripts/08_compare_detectors.py [--detectors yolo26 yolov8 rtdetr] [--matching MODE]

Two comparisons, both on the SAME test images with PAIRED bootstrap resamples:

  at_own_threshold      each detector at its own operating threshold
                        (calibration.json: "max_f1" or the fixed number);
  at_matched_cost       every detector moved to the threshold that gives the
                        same number of false alarms per image (the median of
                        the detectors' own). A ranking difference that only
                        appears in the first comparison is a difference of
                        operating point, not of reliability.

Writes outputs/<experiment>/comparison/<mode>/{ranking.json, detectors.png}.
"""
from __future__ import annotations

import numpy as np
from _common import setup

from confdet.calibration.detection_ece import operating_point
from confdet.evaluation.records import load_records
from confdet.evaluation.rq3_calibration_gap import (
    metric_replicates, ranking_comparison, threshold_for_false_alarms,
)
from confdet.evaluation.stats import bootstrap_image_weights, percentile_ci
from confdet.pipeline import PostHoc, tables_from_records
from confdet.utils.io_utils import load_results, matching_modes, save_results
from confdet.utils.paths import calibration_path, comparison_dir, predictions_path, results_dir
from confdet.visualization import plots


def _entry(T, tau, bins, boot):
    point = operating_point(T, tau, bins)
    reps = metric_replicates(T, tau, bins, boot)
    for k in ("dece", "ma_ece", "miss_mass", "oce_max", "ma_brier"):
        point[f"{k}_ci"] = percentile_ci(reps[k])
    return {"point": point, "reps": reps}


def compare(mode: str, wanted: list[str], cfg: dict, log) -> None:
    ev = cfg["evaluation"]
    bins = int(ev.get("confidence_bins", 10))
    dev = cfg["experiment"]["development"]
    tables, taus, image_ids, boot = {}, {}, None, None
    for det in wanted:
        path, cal = predictions_path(cfg, det, dev, "test"), calibration_path(cfg, det, mode)
        if not path.exists() or not cal.exists():
            log.warning(f"[{mode}] {det}: predictions or calibration missing; skipped")
            continue
        posthoc = PostHoc(load_results(cal))
        T = tables_from_records(cfg, load_records(path), mode, context_conf=posthoc.context_conf)
        if image_ids is None:
            image_ids = T.image_ids
            boot = bootstrap_image_weights(T.n_images, int(ev.get("bootstrap_replicates", 500)),
                                           int(cfg["project"].get("seed", 42)))
        elif T.image_ids != image_ids:
            raise SystemExit(f"{det} was evaluated on different test images than the others; "
                             f"re-run scripts/05_predict.py for every detector after scripts/02.")
        tables[det], taus[det] = T, posthoc.tau
    if not tables:
        log.warning(f"[{mode}] no detector has cached predictions and a calibration yet")
        return

    log.info(f"===== matching: {mode} =====")
    own = {det: _entry(T, taus[det], bins, boot) for det, T in tables.items()}
    for det, e in own.items():
        p = e["point"]
        log.info(f"  own threshold  {det:<8} tau {taus[det]:.2f} | D-ECE {p['dece']:.4f} | MA-ECE {p['ma_ece']:.4f} | "
                 f"OCE {p['oce_max']:.4f} | recall {p['recall']:.3f} | false alarms/image {p['fp_per_image']:.2f}")
    cmp = {"matching": mode, "at_own_threshold": ranking_comparison(own)}
    cmp["at_own_threshold"]["points"] = {d: v["point"] for d, v in own.items()}

    budget = float(np.median([e["point"]["fp_per_image"] for e in own.values()]))
    matched_tau = {det: threshold_for_false_alarms(T, budget) for det, T in tables.items()}
    matched = {det: _entry(T, matched_tau[det], bins, boot) for det, T in tables.items()}
    for det, e in matched.items():
        p = e["point"]
        log.info(f"  matched cost   {det:<8} tau {matched_tau[det]:.2f} | D-ECE {p['dece']:.4f} | MA-ECE "
                 f"{p['ma_ece']:.4f} | OCE {p['oce_max']:.4f} | recall {p['recall']:.3f} | false alarms/image "
                 f"{p['fp_per_image']:.2f}")
    cmp["at_matched_cost"] = ranking_comparison(matched)
    cmp["at_matched_cost"].update({"fp_per_image_budget": budget, "thresholds": matched_tau,
                                   "points": {d: v["point"] for d, v in matched.items()}})

    for key in ("at_own_threshold", "at_matched_cost"):
        block = cmp[key]
        if "rankings" not in block:
            log.info(f"  {key}: {block['note']}")
            continue
        log.info(f"  {key}: " + " | ".join(f"{m}: {' < '.join(o)}" for m, o in block["rankings"].items()))
        for row in block["pairwise"]:
            sep = ", ".join(f"{m} prefers {row[m]['better_if_separated']}" for m in row["separated_by"]) or "none"
            log.info(f"    {row['a']} vs {row['b']}: separated by: {sep}; flips vs D-ECE (both intervals "
                     f"exclude zero, signs differ) {row['flips_vs_dece']}; "
                     f"P(D-ECE and MA-ECE disagree) = {row['prob_dece_and_ma_ece_disagree']:.2f}")
            if not row["dece"]["separates"]:
                log.info(f"      D-ECE cannot separate {row['a']} and {row['b']}")
    flips = {k: cmp[k].get("any_ranking_flip_vs_dece") for k in ("at_own_threshold", "at_matched_cost")}
    cmp["ranking_flip_survives_matched_cost"] = bool(flips["at_own_threshold"] and flips["at_matched_cost"])
    log.info(f"  ranking flip vs D-ECE: own threshold {flips['at_own_threshold']}, matched cost "
             f"{flips['at_matched_cost']}")

    cmp["rq2_headline"] = {}
    for det in tables:
        s = results_dir(cfg, det, mode) / "summary.json"
        if s.exists():
            cmp["rq2_headline"][det] = load_results(s)["headline"].get("rq2")
    out = comparison_dir(cfg, mode)
    save_results(cmp, out / "ranking.json")
    plots.save(plots.plot_detector_comparison(cmp["at_matched_cost"]["points"],
                                              f"Detector reliability at matched cost "
                                              f"({budget:.2f} false alarms/image; matching: {mode})"),
               out / "detectors.png")
    log.info(f"  saved {out / 'ranking.json'}")


def main():
    def extra(p):
        p.add_argument("--detectors", nargs="+", default=None)
        p.add_argument("--matching", default=None)

    args, cfg, log = setup("compare_detectors", __doc__, extra=extra)
    wanted = args.detectors or list(cfg["detectors"]["models"])
    for mode in ([args.matching] if args.matching else matching_modes(cfg)):
        compare(mode, wanted, cfg, log)


if __name__ == "__main__":
    main()
