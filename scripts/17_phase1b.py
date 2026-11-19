#!/usr/bin/env python3
"""Phase 1b (v0.6), CPU only, on cached predictions (class-agnostic).

  python scripts/17_phase1b.py [--detectors yolo26 yolov8 rtdetr yolo26_ens rtdetr_ens]
                               [--seed-families yolo26=yolo26,yolo26_s1,yolo26_s2 rtdetr=rtdetr,rtdetr_s1,rtdetr_s2]

  1. Mondrian conformal on the Gilg IoU-aware score and on our default score, every alpha,
     same calib halves, context edges and minimum group size as mondrian_confidence.
  2. Paired RQ2 differences: Gilg minus our default score, Gilg minus z_dup (same resamples).
  3. LaECE_0 / LaACE_0 of every seed model; family mean, SD, and the RT-DETR - YOLO26 gap.
  4. Strata granularity: (a) current context edges (headline) vs (b) 4 strata at calib o_hat
     quantiles with every stratum >= 300 calib objects (rule fixed before test, in
     evaluation/diagnostics.quantile_strata_edges), mondrian_confidence at every alpha.
  5. Occluder diagnostic for vehicles with o_j >= 0.5 and o_hat < 0.30 (test role).

Writes outputs/<experiment>/<detector>/class_agnostic/results/phase1b.json and
outputs/<experiment>/comparison/class_agnostic/phase1b_seeds_laece.json.
"""
from __future__ import annotations

import numpy as np
from _common import setup

from confdet.evaluation import baselines as bl
from confdet.evaluation.diagnostics import occluder_diagnostic, paired_recovery, quantile_strata_edges
from confdet.evaluation.records import load_records
from confdet.evaluation.stats import bootstrap_image_weights
from confdet.pipeline import PostHoc, _calibrator_half, calibrator_candidate_rows, tables_from_records
from confdet.uq.calibrator import fit_calibrator
from confdet.uq.conformal import fit_thresholds
from confdet.utils.io_utils import load_results, save_results
from confdet.utils.paths import calibration_path, comparison_dir, predictions_path, results_dir

MODE = "class_agnostic"


def _summ(rep: dict) -> dict:
    return {k: rep.get(k) for k in ("coverage", "coverage_ci", "mean_image_fnr", "flags_per_image",
                                    "by_annotated_occlusion", "by_occlusion_context")}


def run(det: str, cfg: dict, log) -> dict:
    ev, ph, occ = cfg["evaluation"], cfg["posthoc"], cfg["occlusion"]
    seed = int(cfg["project"].get("seed", 42))
    dev = cfg["experiment"]["development"]
    edges_o, edges_a = occ["bucket_edges"], list(ph["context_edges"])
    min_group = int(ph.get("min_group_size", 100))
    posthoc = PostHoc(load_results(calibration_path(cfg, det, MODE)))
    tau, ctx_conf = posthoc.tau, posthoc.context_conf
    rs_cal, rs_test = (load_records(predictions_path(cfg, det, dev, r)) for r in ("calib", "test"))
    calib = tables_from_records(cfg, rs_cal, MODE, context_conf=ctx_conf)
    test = tables_from_records(cfg, rs_test, MODE, context_conf=ctx_conf)
    bw = bootstrap_image_weights(test.n_images, int(ev.get("bootstrap_replicates", 500)), seed)
    half = _calibrator_half(calib.image_ids, float(ph.get("calibrator_fraction", 0.5)), seed)
    gm = (~half)[calib.g_img]                                   # objects of the threshold half
    rows = calibrator_candidate_rows(calib, cfg, seed)
    geo_cal, geo_test = bl.candidate_geometry(rs_cal.records), bl.candidate_geometry(rs_test.records)
    gilg = bl.fit_gilg(calib.c_conf[rows], geo_cal["u_higher"][rows], calib.c_tp[rows])
    z_dup = fit_calibrator(calib.c_conf[rows], calib.c_ctx[rows], calib.c_dup[rows], calib.c_tp[rows], "z_dup",
                           duplicate_iou=float(ph.get("duplicate_iou", 0.5)))
    cal_scores = {"gilg_iou_aware": bl.predict_gilg(gilg, calib.c_conf, geo_cal["u_higher"]),
                  "default_score": posthoc.calibrator.predict(calib.c_conf, calib.c_ctx, calib.c_dup)}
    test_scores = {"gilg_iou_aware": bl.predict_gilg(gilg, test.c_conf, geo_test["u_higher"]),
                   "default_score": posthoc.calibrator.predict(test.c_conf, test.c_ctx, test.c_dup),
                   "z_dup": z_dup.predict(test.c_conf, test.c_ctx, test.c_dup)}

    def obj(scores_c, t):
        return np.where(t.g_proposed, scores_c[np.clip(t.g_cand, 0, None)] if t.n_cand else 0.0, 0.0)

    def evaluate(thr, cand_score):
        found, kept = bl.found_for_rule(test, thr, cand_score)
        rep = bl.report_from_found(test, found, kept & (test.c_conf < tau), tau, edges_o, edges_a, bw)
        return {"certifiable": bool(all(g.achievable for g in thr.groups)), **_summ(rep)}

    # 1. Mondrian on other scores (same halves, edges and minimum group size)
    item1, item4 = {}, {}
    strata_b = quantile_strata_edges(calib.g_ctx[gm], 4, 300)
    for a in posthoc.alphas:
        blk = {"mondrian_confidence": evaluate(posthoc.rules(a)["mondrian_confidence"], test.c_conf)}
        for name in ("gilg_iou_aware", "default_score"):
            thr = fit_thresholds(obj(cal_scores[name], calib)[gm], calib.g_proposed[gm], calib.g_ctx[gm], a,
                                 context_edges=edges_a, min_group_size=min_group, score_name=name)
            blk[f"mondrian_{name}"] = evaluate(thr, test_scores[name])
        item1[f"{a:g}"] = blk
        # 4. granularity: (b) quantile strata, scored against both the (b) and the (a) strata
        thr_b = fit_thresholds(calib.g_score[gm], calib.g_proposed[gm], calib.g_ctx[gm], a,
                               context_edges=strata_b["edges"], min_group_size=min_group)
        found, kept = bl.found_for_rule(test, thr_b, test.c_conf)
        flagged = kept & (test.c_conf < tau)
        rep_b = bl.report_from_found(test, found, flagged, tau, edges_o, strata_b["edges"], bw)
        rep_ba = bl.report_from_found(test, found, flagged, tau, edges_o, edges_a, bw)
        item4[f"{a:g}"] = {"a_current": blk["mondrian_confidence"],
                           "b_quantile": {"certifiable": bool(all(g.achievable for g in thr_b.groups)), **_summ(rep_b),
                                          "by_occlusion_context_on_a_strata": rep_ba["by_occlusion_context"]}}

    # 2. paired RQ2 differences
    budgets = ev["extra_fp_per_image_budgets"]
    occ_thr = float(occ.get("occluded_threshold", 0.2))
    item2 = {"vs_default_score": paired_recovery(test, test_scores, "default_score", tau, budgets, bw, occ_thr),
             "vs_z_dup": paired_recovery(test, test_scores, "z_dup", tau, budgets, bw, occ_thr)}

    # 5. occluder diagnostic (test role)
    item5 = occluder_diagnostic(rs_test.records, test, ctx_conf, float(ph.get("duplicate_iou", 0.5)))

    out = {"detector": det, "operating_threshold": tau, "context_conf": ctx_conf,
           "item1_mondrian_on_other_scores": item1, "item2_paired_recovery": item2,
           "item4_strata_granularity": {"rule": "4 strata at calib o_hat quantiles, merge strata < 300 calib objects; "
                                                "fixed before test was looked at", "a_edges": edges_a,
                                        "b": strata_b, "by_alpha": item4},
           "item5_occluder_diagnostic": item5}
    save_results(out, results_dir(cfg, det, MODE) / "phase1b.json")
    a10 = item1["0.1"]
    parts = []
    for k, v in a10.items():
        strata = "/".join("%.2f" % s["coverage"] for s in v["by_occlusion_context"])
        parts.append(f"{k} cov {v['coverage']:.3f} flags {v['flags_per_image']:.2f} o_hat {strata}")
    log.info(f"{det} | a=0.10 | " + " | ".join(parts))
    c = item2["vs_default_score"]["comparisons"]["gilg_iou_aware minus default_score"]
    log.info(f"{det} | Gilg - default: all {c['all_misses']:+.4f} {c['all_misses_ci']} occl {c['occluded_misses']:+.4f} "
             f"{c['occluded_misses_ci']} | strata (b) edges {strata_b['edges']} counts {strata_b['calib_counts']} | "
             f"o_j>=0.5 with o_hat<0.30: {item5['share_with_low_context']:.3f} of {item5['n_heavily_occluded']} -> "
             f"{ {k: round(v, 3) for k, v in item5['shares'].items()} }")
    return out


def seeds_laece(cfg: dict, families: dict, log) -> dict:
    dev = cfg["experiment"]["development"]
    out = {}
    for fam, members in families.items():
        vals = {}
        for m in members:
            cal = calibration_path(cfg, m, MODE)
            if not cal.exists():
                continue
            tau = PostHoc(load_results(cal)).tau
            rs = load_records(predictions_path(cfg, m, dev, "test"))
            t = tables_from_records(cfg, rs, MODE, context_conf=PostHoc(load_results(cal)).context_conf)
            g = bl.candidate_geometry(rs.records)
            k = t.c_conf >= tau
            vals[m] = {"laece0": bl.laece0(t.c_conf[k], g["la_target"][k]),
                       "laace0": bl.laace0(t.c_conf[k], g["la_target"][k]), "threshold": tau}
        arr = {q: np.array([v[q] for v in vals.values()]) for q in ("laece0", "laace0")}
        out[fam] = {"members": vals, **{f"{q}_mean": float(a.mean()) for q, a in arr.items()},
                    **{f"{q}_sd": float(a.std(ddof=1)) if len(a) > 1 else None for q, a in arr.items()},
                    **{f"{q}_range": [float(a.min()), float(a.max())] for q, a in arr.items()}}
        log.info(f"{fam}: LaECE0 {out[fam]['laece0_mean']:.4f} +- {out[fam]['laece0_sd']:.4f} "
                 f"{out[fam]['laece0_range']} | LaACE0 {out[fam]['laace0_mean']:.4f} +- {out[fam]['laace0_sd']:.4f}")
    if {"yolo26", "rtdetr"} <= set(out):
        y, r = out["yolo26"], out["rtdetr"]
        gap = r["laece0_mean"] - y["laece0_mean"]
        out["rtdetr_minus_yolo26"] = {"laece0_gap": gap, "pooled_seed_sd": float(np.sqrt((y["laece0_sd"] ** 2 + r["laece0_sd"] ** 2) / 2)),
                                      "ranges_overlap": not (r["laece0_range"][0] > y["laece0_range"][1]
                                                             or y["laece0_range"][0] > r["laece0_range"][1])}
        log.info(f"RT-DETR - YOLO26 LaECE0 gap {gap:+.4f}; pooled seed SD {out['rtdetr_minus_yolo26']['pooled_seed_sd']:.4f}; "
                 f"ranges overlap: {out['rtdetr_minus_yolo26']['ranges_overlap']}")
    return out


def main():
    def extra(p):
        p.add_argument("--detectors", nargs="+", default=["yolo26", "yolov8", "rtdetr", "yolo26_ens", "rtdetr_ens"])
        p.add_argument("--seed-families", nargs="*",
                       default=["yolo26=yolo26,yolo26_s1,yolo26_s2", "rtdetr=rtdetr,rtdetr_s1,rtdetr_s2"])

    args, cfg, log = setup("phase1b", __doc__, extra=extra)
    for det in args.detectors:
        if calibration_path(cfg, det, MODE).exists():
            run(det, cfg, log)
    fams = {s.split("=")[0]: s.split("=")[1].split(",") for s in args.seed_families}
    save_results(seeds_laece(cfg, fams, log), comparison_dir(cfg, MODE) / "phase1b_seeds_laece.json")


if __name__ == "__main__":
    main()
