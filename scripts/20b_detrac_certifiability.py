#!/usr/bin/env python3
"""Why the RECALIBRATED Mondrian rules of scripts/20 are "NOT certifiable" on UA-DETRAC. Analysis only.

  python scripts/20b_detrac_certifiability.py --config configs/experiment_v06_detrac.yaml [--detectors yolo26 yolov8 rtdetr]

Refits exactly the thresholds scripts/20 fits (same functions, same seed, same context_edges and
min_group_size; nothing changed) for both settings (frozen = BMD-45 calib, recalibrated = UA-DETRAC
calib) and both Mondrian rules (confidence; the pre-specified Gilg IoU-aware score), every alpha, and
records per merged o_hat group: calib n, objects never proposed, the group's smallest certifiable miss
rate (n_unproposed + 1) / (n + 1), and the target alpha. Reads calib predictions only.
Writes outputs/<experiment>/<detector>/class_agnostic/results/ua_detrac_certifiability.json.
"""
from __future__ import annotations

import importlib
from datetime import datetime, timezone

from _common import setup

from confdet.evaluation.records import load_records
from confdet.pipeline import PostHoc, fit_posthoc, tables_from_records
from confdet.uq.conformal import certifiability_by_group
from confdet.utils.io_utils import load_results, save_results
from confdet.utils.paths import calibration_path, predictions_path, results_dir

d20 = importlib.import_module("20_detrac")
DS, MODE = d20.DS, d20.MODE


def run(det, cfg, seed, log):
    dev = cfg["experiment"]["development"]
    posthoc = PostHoc(load_results(calibration_path(cfg, det, MODE)))
    tau, ctx = posthoc.tau, posthoc.context_conf
    rs_bmd = load_records(predictions_path(cfg, det, dev, "calib"))
    t_bmd = tables_from_records(cfg, rs_bmd, MODE, context_conf=ctx)
    rs_c = load_records(predictions_path(cfg, det, DS, "calib"))
    t_c = tables_from_records(cfg, rs_c, MODE, context_conf=ctx)
    refit = PostHoc(fit_posthoc(cfg, t_c, seed, tau=tau, mode=MODE))       # as scripts/20
    out = {"detector": det, "frozen_threshold": tau, "written_at_utc": datetime.now(timezone.utc).isoformat(),
           "note": "thresholds refitted exactly as scripts/20_detrac.py fits them; analysis only, no rule changed",
           "by_alpha": {}}
    for a in posthoc.alphas:
        blk = {}
        for setting, layer, cal_t, cal_rs in (("frozen", posthoc, t_bmd, rs_bmd), ("recalibrated", refit, t_c, rs_c)):
            blk[f"{setting}:mondrian_confidence"] = certifiability_by_group(layer.rules(a)["mondrian_confidence"])
            blk[f"{setting}:mondrian_gilg_iou_aware"] = certifiability_by_group(d20.gilg_rules(cal_t, cal_rs, cfg, a, seed)[1])
        out["by_alpha"][f"{a:g}"] = blk
    save_results(out, results_dir(cfg, det, MODE) / "ua_detrac_certifiability.json")
    for k, rows in out["by_alpha"]["0.1"].items():
        log.info(f"{det} alpha 0.1 {k}: " + " | ".join(
            f"[{r['o_hat_lo']:.2f},{r['o_hat_hi']:.2f}) n {r['n_calib']} unprop {r['n_never_proposed']} "
            f"min_alpha {r['min_certifiable_alpha']:.3f} cert {r['certifiable']}" for r in rows))


def main():
    def extra(p):
        p.add_argument("--detectors", nargs="+", default=["yolo26", "yolov8", "rtdetr"])

    args, cfg, log = setup("detrac_certifiability", __doc__, extra=extra)
    seed = int(cfg["project"].get("seed", 42))
    for det in args.detectors:
        run(det, cfg, seed, log)


if __name__ == "__main__":
    main()
