#!/usr/bin/env python3
"""Phase 1 (v0.6): literature baselines on the cached predictions, next to our rules.

  python scripts/16_baselines.py [--detectors yolo26 yolov8 rtdetr yolo26_ens rtdetr_ens]
                                 [--matching class_agnostic] [--delta 0.05]

Per detector (see evaluation/baselines.py for every definition and citation):

  * certified thresholds, each fitted on the SAME calib images our conformal thresholds
    use (the threshold half of calib; the other half fits the candidate score), at every
    alpha of posthoc.alpha_grid, evaluated once on test:
        crc_fnr (per-image FNR), seqcrc box-count-recall / box-count-threshold,
        pac (delta), our global_confidence, mondrian_confidence (headline, D7) and
        global rule on the duplicate- and context-aware score -- all with coverage
        [CI], per-image FNR [CI], flags/image and coverage by annotated occlusion o_j
        and by occlusion context o_hat;
  * Gilg et al. IoU-aware calibration (bivariate Beta on confidence and IoU with any
    higher-scoring box), fitted on the calib score half (no predictions exist for the
    fit role; the score half is what our own score is fitted on), compared at matched
    extra false alarms (RQ2) with plain confidence, our z_dup and our default score,
    plus the rank correlation of the two duplicate-aware scores;
  * LaECE_0, LaACE_0 (25 bins) and D-ECE at the operating threshold;
  * LRP at the operating threshold and at the LRP-optimal threshold chosen on calib.

Writes outputs/<experiment>/<detector>/<mode>/results/baselines.json and
outputs/<experiment>/comparison/<mode>/baselines_summary.json. No GPU.
"""
from __future__ import annotations

import numpy as np
from _common import setup

from confdet.calibration.detection_ece import dece
from confdet.evaluation import baselines as bl
from confdet.evaluation.records import load_records
from confdet.evaluation.rq2_recovery import recovery_analysis
from confdet.evaluation.stats import bootstrap_image_weights, percentile_ci, spearman
from confdet.pipeline import PostHoc, _calibrator_half, calibrator_candidate_rows, tables_from_records
from confdet.uq.calibrator import fit_calibrator
from confdet.uq.conformal import fit_thresholds
from confdet.utils.io_utils import load_results, matching_modes, save_results
from confdet.utils.paths import calibration_path, comparison_dir, predictions_path, results_dir

OUR_RULES = ("global_confidence", "mondrian_confidence", "global_occlusion_aware")


def run(det: str, mode: str, delta: float, cfg: dict, log) -> dict:
    ev, ph, occ = cfg["evaluation"], cfg["posthoc"], cfg["occlusion"]
    seed = int(cfg["project"].get("seed", 42))
    dev = cfg["experiment"]["development"]
    edges, ctx_edges = occ["bucket_edges"], list(ph["context_edges"])
    occ_thr = float(occ.get("occluded_threshold", 0.2))
    posthoc = PostHoc(load_results(calibration_path(cfg, det, mode)))
    tau, ctx = posthoc.tau, posthoc.context_conf
    rs_cal, rs_test = (load_records(predictions_path(cfg, det, dev, r)) for r in ("calib", "test"))
    calib = tables_from_records(cfg, rs_cal, mode, context_conf=ctx)
    test = tables_from_records(cfg, rs_test, mode, context_conf=ctx)
    bw = bootstrap_image_weights(test.n_images, int(ev.get("bootstrap_replicates", 500)), seed)
    half = _calibrator_half(calib.image_ids, float(ph.get("calibrator_fraction", 0.5)), seed)
    thr_half = bl.subset_images(calib, ~half)                   # what our conformal thresholds are fitted on
    log.info(f"===== {det} | {mode} | tau {tau:.2f} | calib threshold half: {thr_half['n_images']} images, "
             f"{len(thr_half['g_score'])} vehicles | test {test.n_images} images =====")

    # ---------------- Gilg et al. score (needs per-candidate geometry) ----------------
    geo_cal, geo_test = bl.candidate_geometry(rs_cal.records), bl.candidate_geometry(rs_test.records)
    for name, g, t in (("calib", geo_cal, calib), ("test", geo_test, test)):
        if len(g["tp"]) != t.n_cand or not (g["tp"] == t.c_tp).all():
            raise SystemExit(f"{det} {name}: candidate geometry is not aligned with the tables (mode {mode})")
    rows = calibrator_candidate_rows(calib, cfg, seed)
    gilg = bl.fit_gilg(calib.c_conf[rows], geo_cal["u_higher"][rows], calib.c_tp[rows])
    z_dup = fit_calibrator(calib.c_conf[rows], calib.c_ctx[rows], calib.c_dup[rows], calib.c_tp[rows], "z_dup",
                           duplicate_iou=float(ph.get("duplicate_iou", 0.5)))
    score_cal = {"gilg": bl.predict_gilg(gilg, calib.c_conf, geo_cal["u_higher"])}
    score_test = {"confidence": test.c_conf.astype(float),
                  "gilg_iou_aware": bl.predict_gilg(gilg, test.c_conf, geo_test["u_higher"]),
                  "z_dup": z_dup.predict(test.c_conf, test.c_ctx, test.c_dup)}
    if posthoc.calibrator is not None:
        score_test["default_score"] = posthoc.calibrator.predict(test.c_conf, test.c_ctx, test.c_dup)
    rq2 = recovery_analysis(test, score_test, tau, ev["extra_fp_per_image_budgets"], bw, occ_thr)
    weak = test.c_conf < tau
    gilg_vs = {
        "spearman_gilg_vs_z_dup_weak_candidates": spearman(score_test["gilg_iou_aware"][weak], score_test["z_dup"][weak]),
        "spearman_gilg_vs_confidence_weak_candidates": spearman(score_test["gilg_iou_aware"][weak], test.c_conf[weak]),
        "spearman_u_higher_vs_dup": spearman(geo_test["u_higher"], test.c_dup),
        "recovery": {k: {f: v.get(f) for f in ("mean_recovery_over_budget_range", "mean_recovery_ci",
                                               "mean_recovery_delta_vs_confidence", "mean_recovery_delta_ci",
                                               "mean_recovery_occluded_delta_vs_confidence",
                                               "mean_recovery_occluded_delta_ci")}
                     for k, v in rq2["policies"].items()},
        "gilg_model": gilg, "fitted_on": "calib score half (the rows our own candidate score is fitted on)"}

    # ---------------- certified thresholds at every alpha ----------------
    gm, cm = thr_half["g_mask"], thr_half["c_mask"]
    obj_gilg_cal = np.where(calib.g_proposed, score_cal["gilg"][np.clip(calib.g_cand, 0, None)], 0.0)
    by_alpha = {}
    for a in posthoc.alphas:
        rules = {}
        fits = {"crc_fnr": bl.crc_fnr_threshold(thr_half["g_score"], thr_half["g_proposed"], thr_half["g_img"],
                                                thr_half["n_images"], a),
                "seqcrc_box_count_recall": bl.seqcrc_confidence(thr_half["g_img"], thr_half["n_images"],
                                                                thr_half["c_img"], thr_half["c_conf"], a,
                                                                "box_count_recall"),
                "seqcrc_box_count_threshold": bl.seqcrc_confidence(thr_half["g_img"], thr_half["n_images"],
                                                                   thr_half["c_img"], thr_half["c_conf"], a,
                                                                   "box_count_threshold"),
                "pac": bl.pac_threshold(thr_half["g_score"], thr_half["g_proposed"], a, delta)}
        for name, fit in fits.items():
            rep = bl.threshold_rule_report(test, fit["threshold"], tau, edges, ctx_edges, bw)
            rules[name] = {"calibration": fit, **rep}
        rules_ours = posthoc.rule_inputs(test, a)
        for name in OUR_RULES:
            if name not in rules_ours:
                continue
            thr, cand_score = rules_ours[name]
            found, kept = bl.found_for_rule(test, thr, cand_score)
            rep = bl.report_from_found(test, found, kept & (test.c_conf < tau), tau, edges, ctx_edges, bw)
            rules[name] = {"calibration": {"certifiable": bool(all(g.achievable for g in thr.groups)),
                                           "groups": [{"threshold": None if not np.isfinite(g.threshold) else float(g.threshold),
                                                       "achievable": bool(g.achievable), "n": int(g.n)} for g in thr.groups]},
                           "achievable": True, **rep}
        g_thr = fit_thresholds(obj_gilg_cal[gm], calib.g_proposed[gm], None, a, score_name="gilg")
        found, kept = bl.found_for_rule(test, g_thr, score_test["gilg_iou_aware"])
        rules["global_gilg_iou_aware"] = {"calibration": {"certifiable": bool(all(g.achievable for g in g_thr.groups))},
                                          "achievable": True,
                                          **bl.report_from_found(test, found, kept & (test.c_conf < tau), tau, edges,
                                                                 ctx_edges, bw)}
        by_alpha[f"{a:g}"] = {"target_coverage": 1 - a, "rules": rules}
        r = rules
        log.info(f"  alpha {a:g}: " + " | ".join(
            f"{k} {v['coverage']:.3f} ({v['flags_per_image']:.1f} fl/img)" if v.get("achievable") else f"{k} n/c"
            for k, v in r.items()))

    # ---------------- localisation-aware calibration and LRP ----------------
    keep = test.c_conf >= tau
    kidx = np.nonzero(keep)[0]
    conf_k, la_k, tp_k, img_k = test.c_conf[kidx], geo_test["la_target"][kidx], test.c_tp[kidx], test.c_img[kidx]
    nb = min(200, len(bw))
    calib_metrics = {
        "at_threshold": tau,
        "laece0": bl.laece0(conf_k, la_k), "laece0_ci": bl.bootstrap_rows(lambda i: bl.laece0(conf_k[i], la_k[i]), img_k, bw[:nb]),
        "laace0": bl.laace0(conf_k, la_k), "laace0_ci": bl.bootstrap_rows(lambda i: bl.laace0(conf_k[i], la_k[i]), img_k, bw[:nb]),
        "dece": dece(conf_k, tp_k, int(ev.get("confidence_bins", 10))),
        "n_bins_la": 25, "bootstrap_replicates": nb}
    olrp = bl.olrp_threshold(calib, geo_cal)
    lrp_rows = {}
    for label, t in (("operating_threshold_max_f1", tau), ("lrp_optimal_threshold_from_calib", olrp["threshold"])):
        pt = bl.lrp_at(test, geo_test, t)
        reps = [bl.lrp_at(test, geo_test, t, w_img=w)["lrp"] for w in bw[:nb]]
        n_det = pt["n_tp"] + pt["n_fp"]
        lrp_rows[label] = {"threshold": float(t), **pt, "lrp_ci": percentile_ci(reps),
                           "recall": pt["n_tp"] / (pt["n_tp"] + pt["n_fn"]),
                           "precision": pt["n_tp"] / n_det if n_det else None,
                           "fp_per_image": pt["n_fp"] / test.n_images,
                           "strata": bl.threshold_rule_report(test, t, t, edges, ctx_edges, bw)}
    lrp_rows["olrp_on_calib"] = olrp["olrp"]
    log.info(f"  LaECE0 {calib_metrics['laece0']:.4f} | LaACE0 {calib_metrics['laace0']:.4f} | D-ECE "
             f"{calib_metrics['dece']:.4f} | LRP at tau {lrp_rows['operating_threshold_max_f1']['lrp']:.4f} | "
             f"LRP-optimal threshold {olrp['threshold']:.2f} -> test LRP "
             f"{lrp_rows['lrp_optimal_threshold_from_calib']['lrp']:.4f}")
    log.info(f"  Gilg vs z_dup: Spearman {gilg_vs['spearman_gilg_vs_z_dup_weak_candidates']:.3f}; RQ2 vs confidence: " +
             " | ".join(f"{k} {v['mean_recovery_delta_vs_confidence']:+.4f}" for k, v in gilg_vs["recovery"].items()
                        if v.get("mean_recovery_delta_vs_confidence") is not None))
    out = {"detector": det, "matching": mode, "operating_threshold": tau, "delta": delta,
           "calibrated_on": {"images": thr_half["n_images"], "vehicles": int(len(thr_half["g_score"])),
                             "note": "calib threshold half, the same images our conformal thresholds use"},
           "by_alpha": by_alpha, "gilg_vs_ours": gilg_vs, "calibration_metrics": calib_metrics, "lrp": lrp_rows}
    save_results(out, results_dir(cfg, det, mode) / "baselines.json")
    return out


def main():
    def extra(p):
        p.add_argument("--detectors", nargs="+", default=["yolo26", "yolov8", "rtdetr", "yolo26_ens", "rtdetr_ens"])
        p.add_argument("--matching", default=None)
        p.add_argument("--delta", type=float, default=0.05)

    args, cfg, log = setup("baselines", __doc__, extra=extra)
    mode = args.matching or matching_modes(cfg)[0]
    summary = {}
    for det in args.detectors:
        if not calibration_path(cfg, det, mode).exists():
            log.warning(f"{det}: no calibration for {mode}; skipped")
            continue
        r = run(det, mode, args.delta, cfg, log)
        summary[det] = {a: {k: {f: v.get(f) for f in ("threshold", "achievable", "coverage", "coverage_ci",
                                                      "mean_image_fnr", "mean_image_fnr_ci", "flags_per_image")}
                            for k, v in blk["rules"].items()} for a, blk in r["by_alpha"].items()}
    save_results({"matching": mode, "detectors": summary}, comparison_dir(cfg, mode) / "baselines_summary.json")
    log.info(f"Saved {comparison_dir(cfg, mode) / 'baselines_summary.json'}")


if __name__ == "__main__":
    main()
