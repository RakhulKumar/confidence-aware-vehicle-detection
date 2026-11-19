#!/usr/bin/env python3
"""UA-DETRAC (v0.6 Phase 3c-3d) with the CURRENT models, class-agnostic. CPU only (needs the cached
UA-DETRAC predictions from scripts/05 with configs/experiment_detrac.yaml).

  python scripts/20_detrac.py --config configs/experiment_detrac.yaml [--detectors yolo26 yolov8 rtdetr]

3c  proxy validation on the test frames: Spearman of our box-overlap proxy (vehicle boxes only,
    bottom-edge rule) and of the deployable occlusion context o_hat (per detector) against the
    annotated per-vehicle occlusion -- headline reading (occ_vehicle) and naive reading (occ_naive) --
    with sequence-bootstrap intervals; non-vehicle occlusion and truncation summarised separately.
3d  zero-shot transfer: recall at the frozen operating threshold; conformal rules FROZEN (fitted on
    BMD-45 calib) and RECALIBRATED (re-fitted on UA-DETRAC calib = train sequences): global_confidence,
    mondrian_confidence and the pre-specified headline Mondrian on the Gilg IoU-aware score; coverage
    overall, by annotated occlusion (occ_vehicle buckets) and by o_hat stratum; flags per image; every
    alpha of the grid. Intervals resample SEQUENCES.
Writes outputs/<experiment>/<detector>/class_agnostic/results/ua_detrac_transfer.json and
outputs/<experiment>/comparison/class_agnostic/ua_detrac_proxy_validation.json.
"""
from __future__ import annotations

import copy
import json

import numpy as np
from _common import setup

from confdet.data.detrac import CLASSES, annotations_for_tables, group_bootstrap_image_weights
from confdet.data.occlusion import occlusion_severity_array
from confdet.evaluation import baselines as bl
from confdet.evaluation.records import load_records
from confdet.evaluation.stats import bootstrap_mean_ci, percentile_ci, spearman
from confdet.pipeline import PostHoc, _calibrator_half, calibrator_candidate_rows, fit_posthoc, tables_from_records
from confdet.uq.conformal import fit_thresholds
from confdet.utils.io_utils import load_results, save_results
from confdet.utils.paths import calibration_path, comparison_dir, predictions_path, processed_root, results_dir

DS, MODE = "ua_detrac", "class_agnostic"


def vehicle_proxy(records) -> np.ndarray:
    """Box-overlap occlusion among vehicle boxes only (ignore regions are not occluders), for the
    car / van / bus objects, in table order."""
    out = []
    for r in records:
        cls = np.asarray(r.gt_cls)
        veh = cls != CLASSES.index("ignore")
        sev = occlusion_severity_array(np.asarray(r.gt_xyxy, float)[veh])
        keep = np.isin(cls[veh], (0, 1, 2))
        out += sev[keep].tolist()
    return np.asarray(out, float)


def seq_of(image_ids):
    return [i.rsplit("_img", 1)[0] for i in image_ids]


def rho_ci(x, y, groups, n=300, seed=0):
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    idx_of = {g: np.nonzero(groups == g)[0] for g in uniq}
    reps = []
    for _ in range(n):
        rows = np.concatenate([idx_of[g] for g in rng.choice(uniq, len(uniq))])
        reps.append(spearman(x[rows], y[rows]))
    return spearman(x, y), percentile_ci(reps)


def gilg_rules(calib_tables, calib_records, cfg, alpha, seed):
    """Gilg score fitted on the calib score half, Mondrian thresholds on the calib threshold half."""
    ph = cfg["posthoc"]
    geo = bl.align_to_tables(np.concatenate([bl.candidate_geometry([r])["u_higher"] for r in calib_records.records]),
                             calib_records.records, calib_tables)
    rows = calibrator_candidate_rows(calib_tables, cfg, seed)
    model = bl.fit_gilg(calib_tables.c_conf[rows], geo[rows], calib_tables.c_tp[rows])
    s = bl.predict_gilg(model, calib_tables.c_conf, geo)
    half = _calibrator_half(calib_tables.image_ids, float(ph.get("calibrator_fraction", 0.5)), seed)
    gm = (~half)[calib_tables.g_img]
    obj = np.where(calib_tables.g_proposed, s[np.clip(calib_tables.g_cand, 0, None)] if calib_tables.n_cand else 0.0, 0.0)
    thr = fit_thresholds(obj[gm], calib_tables.g_proposed[gm], calib_tables.g_ctx[gm], alpha,
                         context_edges=list(ph["context_edges"]), min_group_size=int(ph.get("min_group_size", 100)),
                         score_name="gilg_iou_aware")
    return model, thr


def run(det, cfg, seed, log, proxy_rows):
    ph = cfg["posthoc"]
    dev = cfg["experiment"]["development"]
    root = processed_root(cfg)
    posthoc = PostHoc(load_results(calibration_path(cfg, det, MODE)))
    tau, ctx = posthoc.tau, posthoc.context_conf
    rs_bmd = load_records(predictions_path(cfg, det, dev, "calib"))
    t_bmd = tables_from_records(cfg, rs_bmd, MODE, context_conf=ctx)
    rs_c, rs_t = (load_records(predictions_path(cfg, det, DS, r)) for r in ("calib", "test"))
    t_c = tables_from_records(cfg, rs_c, MODE, context_conf=ctx)
    t_t = tables_from_records(cfg, rs_t, MODE, context_conf=ctx)
    ann = json.loads((root / DS / "annotations_test.json").read_text(encoding="utf-8"))
    occ = {k: annotations_for_tables(ann, rs_t.records, t_t, k)
           for k in ("occ_vehicle", "occ_naive", "occ_nonvehicle", "truncation")}
    groups = seq_of(t_t.image_ids)
    bw = group_bootstrap_image_weights(groups, int(cfg["evaluation"].get("bootstrap_replicates", 500)), seed)
    t_eval = copy.copy(t_t)
    t_eval.g_occ = occ["occ_vehicle"]                       # coverage by ANNOTATED occlusion (headline reading)
    g_groups = [groups[i] for i in t_t.g_img]
    proxy_rows[det] = {f"o_hat_vs_{k}": rho_ci(t_t.g_ctx, occ[k], g_groups) for k in ("occ_vehicle", "occ_naive")}
    found = t_t.g_proposed & (t_t.g_score >= tau)
    out = {"detector": det, "frozen_threshold": tau, "n_images": t_t.n_images, "n_vehicles": t_t.n_gt,
           "n_sequences": len(set(groups)),
           "recall_at_frozen_threshold": float(found.mean()),
           "recall_ci_sequence_bootstrap": bootstrap_mean_ci(found.astype(float), t_t.g_img, bw),
           "recall_at_floor": float(t_t.g_proposed.mean()),
           "fp_per_image_at_frozen_threshold": float(((t_t.c_conf >= tau) & ~t_t.c_tp).sum() / t_t.n_images),
           "by_alpha": {}}
    edges_o, edges_a = cfg["occlusion"]["bucket_edges"], list(ph["context_edges"])
    refit = PostHoc(fit_posthoc(cfg, t_c, seed, tau=tau, mode=MODE))
    geo_t = bl.align_to_tables(np.concatenate([bl.candidate_geometry([r])["u_higher"] for r in rs_t.records]),
                               rs_t.records, t_t)
    for a in posthoc.alphas:
        blk = {}
        for setting, ph_layer, cal_t, cal_rs in (("frozen", posthoc, t_bmd, rs_bmd), ("recalibrated", refit, t_c, rs_c)):
            rules = ph_layer.rule_inputs(t_t, a)
            for name in ("global_confidence", "mondrian_confidence"):
                thr, cs = rules[name]
                f, kept = bl.found_for_rule(t_eval, thr, cs)
                blk[f"{setting}:{name}"] = {"certifiable": bool(all(g.achievable for g in thr.groups)),
                                            **bl.report_from_found(t_eval, f, kept & (t_t.c_conf < tau), tau, edges_o, edges_a, bw)}
            gm, gthr = gilg_rules(cal_t, cal_rs, cfg, a, seed)
            f, kept = bl.found_for_rule(t_eval, gthr, bl.predict_gilg(gm, t_t.c_conf, geo_t))
            blk[f"{setting}:mondrian_gilg_iou_aware"] = {"certifiable": bool(all(g.achievable for g in gthr.groups)),
                                                       **bl.report_from_found(t_eval, f, kept & (t_t.c_conf < tau), tau, edges_o, edges_a, bw)}
        out["by_alpha"][f"{a:g}"] = blk
    save_results(out, results_dir(cfg, det, MODE) / "ua_detrac_transfer.json")
    r = out["by_alpha"]["0.1"]
    log.info(f"{det}: recall {out['recall_at_frozen_threshold']:.3f} {out['recall_ci_sequence_bootstrap']} at tau {tau:.2f} | "
             + " | ".join(f"{k} {v['coverage']:.3f} ({v['flags_per_image']:.2f} fl/img, cert {v['certifiable']})" for k, v in r.items()))
    return occ


def main():
    def extra(p):
        p.add_argument("--detectors", nargs="+", default=["yolo26", "yolov8", "rtdetr"])

    args, cfg, log = setup("detrac", __doc__, extra=extra)
    seed = int(cfg["project"].get("seed", 42))
    proxy_rows, occ = {}, None
    for det in args.detectors:
        occ = run(det, cfg, seed, log, proxy_rows)
    rs_t = load_records(predictions_path(cfg, args.detectors[0], DS, "test"))
    t_t = tables_from_records(cfg, rs_t, MODE, context_conf=0.5)
    prox = vehicle_proxy(rs_t.records)
    groups = [seq_of(t_t.image_ids)[i] for i in t_t.g_img]
    summary = {"n_vehicles": int(len(prox)),
               "box_overlap_proxy_vs_occ_vehicle": rho_ci(prox, occ["occ_vehicle"], groups),
               "box_overlap_proxy_vs_occ_naive": rho_ci(prox, occ["occ_naive"], groups),
               "o_hat_per_detector": proxy_rows,
               "share_vehicles_occluded_by_vehicle_ge_0.2": float((occ["occ_vehicle"] >= 0.2).mean()),
               "share_occluded_by_nonvehicle_gt_0": float((occ["occ_nonvehicle"] > 0).mean()),
               "share_truncated_gt_0": float((occ["truncation"] > 0).mean()),
               "annotated_occlusion_quantiles": np.quantile(occ["occ_vehicle"], [0.5, 0.75, 0.9, 0.99]).tolist()}
    save_results(summary, comparison_dir(cfg, MODE) / "ua_detrac_proxy_validation.json")
    log.info(f"proxy validation: box-overlap vs occ_vehicle {summary['box_overlap_proxy_vs_occ_vehicle']} | vs naive "
             f"{summary['box_overlap_proxy_vs_occ_naive']} | o_hat: {proxy_rows}")


if __name__ == "__main__":
    main()
