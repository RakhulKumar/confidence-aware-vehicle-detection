#!/usr/bin/env python3
"""Stage 4: answer RQ1-RQ5 for one detector from cached predictions, once per
matching mode (evaluation.matching_modes; the first is the primary one).

  python scripts/07_run_evaluation.py [--detector NAME] [--matching MODE]

Reads   outputs/<experiment>/<detector>/predictions/*.npz
        outputs/<experiment>/<detector>/<mode>/calibration.json
Writes  outputs/<experiment>/<detector>/<mode>/results/{detection,rq1,rq2,
        conformal,rq3,rq4_<dataset>,rq5,summary}.json  and  .../figures/*.png

No GPU. Every interval is an image-level bootstrap interval. The summary
starts with the gates (is the detector usable here; is there enough
occlusion to say anything) and a result that fails a gate is labelled so.

RQ5 (why vehicles are missed) needs class labels and is always computed with
native class-aware matching, whatever the mode.
"""
from __future__ import annotations

from _common import setup

from confdet.data.splits import has_role, processed_info
from confdet.evaluation.detection_metrics import detection_summary
from confdet.evaluation.records import load_records
from confdet.evaluation.rq1_correlation import occlusion_analysis
from confdet.evaluation.rq2_recovery import conformal_report, policy_scores, recovery_analysis
from confdet.evaluation.rq3_calibration_gap import calibration_analysis
from confdet.evaluation.rq4_cross_dataset import transfer_summary
from confdet.evaluation.rq5_miss_causes import miss_cause_analysis
from confdet.evaluation.stats import bootstrap_image_weights
from confdet.pipeline import (
    PostHoc, basic_bundle, fit_posthoc, tables_from_records, train_class_counts, transfer_mode,
    validity_curve,
)
from confdet.utils.io_utils import load_results, matching_modes, save_results
from confdet.utils.paths import calibration_path, figures_dir, predictions_path, processed_root, results_dir
from confdet.visualization import plots


def _fmt_ci(ci):
    return "[n/a]" if not ci or ci[0] is None or ci[0] != ci[0] else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _f(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def main():
    args, cfg, log = setup("run_evaluation", __doc__, detector=True,
                           extra=lambda p: p.add_argument("--matching", default=None))
    for mode in ([args.matching] if args.matching else matching_modes(cfg)):
        evaluate(args.detector, mode, cfg, log)


def evaluate(det: str, mode: str, cfg: dict, log) -> None:
    ev, occ = cfg["evaluation"], cfg["occlusion"]
    edges, occ_thr = occ["bucket_edges"], float(occ.get("occluded_threshold", 0.2))
    n_boot, seed = int(ev.get("bootstrap_replicates", 500)), int(cfg["project"].get("seed", 42))
    dev = cfg["experiment"]["development"]
    root = processed_root(cfg)
    res, figs = results_dir(cfg, det, mode), figures_dir(cfg, det, mode)
    names = processed_info(dev, root).names
    posthoc = PostHoc(load_results(calibration_path(cfg, det, mode)))
    tau, ctx = posthoc.tau, posthoc.context_conf
    summary = {"experiment": cfg["experiment"]["name"], "detector": det, "development_dataset": dev,
               "matching": mode, "standard_threshold": tau, "gates": {}, "headline": {}}

    # ------------------------------------------------------------------ development test role
    rs = load_records(predictions_path(cfg, det, dev, "test"))
    T = tables_from_records(cfg, rs, mode, context_conf=ctx)
    T_native = T if mode == "native" else tables_from_records(cfg, rs, "native", context_conf=ctx)
    T_agn = T if mode == "class_agnostic" else tables_from_records(cfg, rs, "class_agnostic", context_conf=ctx)
    bw = bootstrap_image_weights(T.n_images, n_boot, seed)
    log.info(f"===== {det} | matching: {mode} | operating threshold {tau:.2f} =====")
    log.info(f"[{dev}/test] {T.n_images} images, {T.n_gt} vehicles, {T.n_cand} raw candidates")

    mode_names = {"native": names, "coarse": ev.get("common_classes")}.get(mode)
    bundle = basic_bundle(cfg, T, mode_names, posthoc, bw, tau)
    d = bundle["detection"]
    save_results({"matching": mode, "standard_threshold": tau, "this_mode": d, "gate": bundle["gate"],
                  "class_aware": detection_summary(T_native, tau, names),
                  "class_agnostic": detection_summary(T_agn, tau)}, res / "detection.json")
    log.info(f"  detection: AP50 {_f(d['ap50'])} | recall@{tau:.2f} {_f(d['recall_at_standard'])} | "
             f"precision {_f(d['precision_at_standard'])} | false alarms/image "
             f"{_f(d['fp_per_image_at_standard'], 2)} | recall at floor {_f(d['recall_at_floor'])}")
    summary["headline"]["detection"] = {k: d[k] for k in (
        "ap50", "recall_at_standard", "recall_at_floor", "precision_at_standard", "fp_per_image_at_standard")}
    summary["gates"]["detector_usable_on_development"] = bundle["gate"]
    if not bundle["gate"]["usable"]:
        log.error(f"  GATE FAILED: {bundle['gate']['message']}")

    # RQ1
    rq1 = occlusion_analysis(T, tau, edges, bw, occ_thr, int(ev.get("min_occluded_objects", 200)),
                             int(ev.get("min_bucket_n", 30)), seed=seed)
    save_results(rq1, res / "rq1.json")
    plots.save(plots.plot_rq1(rq1), figs / "rq1_occlusion_vs_misses.png")
    summary["gates"]["enough_occlusion"] = rq1["power"]
    powered = rq1["power"]["adequate"]
    if not powered:
        log.warning(f"  POWER GATE: {rq1['power']['message']}")
    m = rq1["logistic_miss_model"]
    log.info("  RQ1 per bucket (n, miss rate): " + "  ".join(
        f"[{b['lo']:.1f}-{b['hi']:.1f}) n={b['n']} {b.get('miss_rate', float('nan')):.3f}{'' if b['reliable'] else '*'}"
        for b in rq1["buckets"]))
    log.info(f"  RQ1 Spearman {rq1['spearman_occlusion_uncertainty']:.3f} {_fmt_ci(rq1['spearman_ci'])}, "
             f"partial (size, class) {rq1['partial_spearman_given_size_and_class']:.3f} "
             f"{_fmt_ci(rq1['partial_spearman_ci'])}"
             + (f"; odds of a miss x{m['odds_ratio_per_0.1_occlusion']:.2f} per +0.1 occlusion "
                f"{_fmt_ci(m['odds_ratio_ci'])}" if m.get("fitted") else ""))
    summary["headline"]["rq1"] = {
        "interpretable": powered,
        "spearman": rq1["spearman_occlusion_uncertainty"], "spearman_ci": rq1["spearman_ci"],
        "partial_spearman": rq1["partial_spearman_given_size_and_class"],
        "partial_spearman_ci": rq1["partial_spearman_ci"],
        "odds_ratio_per_0.1_occlusion": m.get("odds_ratio_per_0.1_occlusion"),
        "odds_ratio_ci": m.get("odds_ratio_ci"),
        "miss_rate_by_bucket": [b.get("miss_rate") for b in rq1["buckets"]],
        "n_by_bucket": [b["n"] for b in rq1["buckets"]],
        "context_tracks_annotated_occlusion": rq1["spearman_occlusion_vs_deployable_context"],
    }

    # RQ2 -- recovery at matched cost, and conformal validity
    scores = policy_scores(T, posthoc.rules().get("mondrian_confidence"), posthoc.calibrator)
    rq2 = recovery_analysis(T, scores, tau, ev["extra_fp_per_image_budgets"], bw, occ_thr)
    save_results(rq2, res / "rq2.json")
    plots.save(plots.plot_rq2(rq2), figs / "rq2_recovery.png")
    conf = bundle["conformal"]
    curve = validity_curve(posthoc, T, tau, edges, bw)
    save_results({"alpha": posthoc.alpha, "report": conf, "validity_curve": curve,
                  "calibration_notes": posthoc.raw.get("notes", [])}, res / "conformal.json")
    plots.save(plots.plot_coverage(conf), figs / "conformal_coverage_by_occlusion.png")
    plots.save(plots.plot_validity(curve), figs / "conformal_validity.png")
    log.info(f"  RQ2 {rq2['n_missed_at_standard']} misses at the standard threshold "
             f"({rq2['n_missed_occluded']} occluded); recoverable by any post-hoc rule: "
             f"{_f(rq2['recoverable_ceiling'])} of them")
    head2 = {"interpretable_for_occluded": powered, "n_missed": rq2["n_missed_at_standard"],
             "n_missed_occluded": rq2["n_missed_occluded"], "recoverable_ceiling": rq2["recoverable_ceiling"],
             "policies": {}}
    for pol, e in rq2.get("policies", {}).items():
        head2["policies"][pol] = {k: e.get(k) for k in (
            "mean_recovery_over_budget_range", "mean_recovery_ci", "mean_recovery_occluded",
            "mean_recovery_delta_vs_confidence", "mean_recovery_delta_ci", "prob_better_than_confidence",
            "mean_recovery_occluded_delta_vs_confidence", "mean_recovery_occluded_delta_ci",
            "prob_better_occluded")}
        extra = ""
        if "mean_recovery_delta_vs_confidence" in e:
            extra = (f" | vs baseline {e['mean_recovery_delta_vs_confidence']:+.3f} "
                     f"{_fmt_ci(e['mean_recovery_delta_ci'])}, P(better)={e['prob_better_than_confidence']:.2f}")
        log.info(f"    {pol:<16} mean recovery {e['mean_recovery_over_budget_range']:.3f}{extra}")
    summary["headline"]["rq2"] = head2
    head_c = {"alpha": posthoc.alpha, "target": 1 - posthoc.alpha}
    log.info(f"  Conformal rules, target recall {1 - posthoc.alpha:.2f} "
             f"(rule alone = what is guaranteed; system = accepted + flagged):")
    for rule, rep in conf.items():
        sys_rows = rep.get("system_by_occlusion_bucket", rep["by_occlusion_bucket"])
        top = sys_rows[-1]
        head_c[rule] = {"rule_recall": rep["coverage"], "rule_recall_ci": rep["coverage_ci"],
                        "system_recall": rep.get("system_recall", rep["coverage"]),
                        "system_recall_most_occluded_bucket": top["coverage"],
                        "n_most_occluded_bucket": top["n"],
                        "all_groups_certifiable": rep.get("all_groups_achievable"),
                        "flagged_per_image": rep.get("flagged_per_image"),
                        "flagged_false_per_image": rep.get("flagged_false_per_image")}
        if rule == "standard_threshold":
            log.info(f"    {rule:<24} recall {_f(rep['coverage'])} {_fmt_ci(rep['coverage_ci'])} | most occluded "
                     f"bucket {_f(top['coverage'])} (n={top['n']})")
            continue
        log.info(f"    {rule:<24} rule alone {_f(rep['coverage'])} {_fmt_ci(rep['coverage_ci'])} | system "
                 f"{_f(rep['system_recall'])} | most occluded bucket {_f(top['coverage'])} | flags/image "
                 f"{_f(rep['flagged_per_image'], 2)} ({_f(rep['flagged_false_per_image'], 2)} false)"
                 + ("" if rep["all_groups_achievable"] else " | target NOT certifiable in some group")
                 + ("" if rep["adds_flags"] else " | adds nothing: the detector already exceeds this target"))
    summary["headline"]["conformal"] = head_c

    # RQ3
    rq3 = calibration_analysis(T, tau, ev["sweep_thresholds"], int(ev.get("confidence_bins", 10)), edges, bw,
                               int(ev.get("min_bucket_n", 30)))
    save_results(rq3, res / "rq3.json")
    plots.save(plots.plot_rq3(rq3), figs / "rq3_calibration_gap.png")
    plots.save(plots.plot_reliability(rq3["reliability_bins"]), figs / "reliability_diagram.png")
    p = rq3["at_standard_threshold"]
    pref = rq3["threshold_each_metric_prefers"]
    log.info(f"  RQ3 D-ECE {p['dece']:.4f} {_fmt_ci(p['dece_ci'])} | MA-ECE {p['ma_ece']:.4f} "
             f"{_fmt_ci(p['ma_ece_ci'])} | miss mass {p['miss_mass']:.4f} | OCE {p['oce_max']:.4f}")
    log.info("  RQ3 threshold each metric prefers: " + ", ".join(
        f"{k} -> {v['threshold']:g} (recall {v['recall_there']:.2f})" for k, v in pref.items()))
    summary["headline"]["rq3"] = {
        "dece": p["dece"], "dece_ci": p["dece_ci"], "ma_ece": p["ma_ece"], "ma_ece_ci": p["ma_ece_ci"],
        "gap": p["gap"], "gap_ci": p["gap_ci"], "miss_mass": p["miss_mass"], "oce_max": p["oce_max"],
        "threshold_preferred": {k: v["threshold"] for k, v in pref.items()},
        "by_bucket_interpretable": powered,
        "share_of_objects_by_bucket": [b["share_of_objects"] for b in rq3["gap_decomposition_by_occlusion"]],
        "share_of_gap_by_bucket": [b["share_of_gap"] for b in rq3["gap_decomposition_by_occlusion"]],
    }

    # RQ5
    if has_role(dev, "fit", root):
        rq5 = miss_cause_analysis(T_native, tau, train_class_counts(dev, len(names), root), edges, bw, names)
        save_results(rq5, res / "rq5.json")
        if rq5.get("cause_shares_by_occlusion_bucket"):
            plots.save(plots.plot_rq5(rq5), figs / "rq5_miss_causes.png")
        log.info(f"  RQ5 causes of {rq5['n_missed']} misses: {rq5['cause_counts']} | verdict {rq5.get('verdict')}")
        summary["headline"]["rq5"] = {"interpretable": powered, "cause_shares": rq5["cause_shares"],
                                      "verdict": rq5.get("verdict"),
                                      "visibility_model": rq5.get("visibility_miss_model"),
                                      "confusion_model": rq5.get("confusion_miss_model")}

    # ------------------------------------------------------------------ RQ4: transfer datasets
    summary["headline"]["rq4"] = {}
    for name in cfg["experiment"].get("transfer") or []:
        path = predictions_path(cfg, det, name, "test")
        if not path.exists():
            log.warning(f"[{name}] no cached predictions ({path}); skipping RQ4 for it")
            continue
        rs_t = load_records(path)
        tmode = transfer_mode(mode)
        out = {"dataset": name, "modes": {},
               "note": "coarse: detector and dataset classes mapped by name to evaluation.common_classes; "
                       "class_agnostic: any vehicle box counts. Development reference numbers use the same "
                       "mode and the same class maps. 'frozen' = detector and post-hoc layer exactly as "
                       "fitted on the development dataset; 'recalibrated' = thresholds re-fitted on this "
                       "dataset's calib role, detector untouched. If this dataset's annotations are "
                       "incomplete, only recall-side quantities (recall, conformal coverage) are "
                       "meaningful on it; precision, false alarms, D-ECE and MA-ECE are not."}
        Tt = tables_from_records(cfg, rs_t, tmode, context_conf=ctx)
        bwt = bootstrap_image_weights(Tt.n_images, n_boot, seed)
        min_n = int(ev.get("min_bucket_n", 30))
        T_dev_common = tables_from_records(cfg, rs, tmode, common=True, context_conf=ctx)
        dev_ref = basic_bundle(cfg, T_dev_common, None, posthoc, bw, tau)
        # RQ4 conformal rows at the headline alpha AND at each alpha in evaluation.rq4_alphas
        # (default: headline and 0.10), wherever thresholds were calibrated for that alpha
        rq4_alphas = sorted({float(a) for a in ev.get("rq4_alphas", [posthoc.alpha, 0.10])
                             if float(a) in posthoc.alphas} | {posthoc.alpha})
        dev_ref["conformal_by_alpha"] = {
            f"{a:g}": conformal_report(T_dev_common, posthoc.rule_inputs(T_dev_common, a), tau, edges, bw, min_n)
            for a in rq4_alphas}
        tb = basic_bundle(cfg, Tt, None, None, bwt, tau)
        tb["conformal_frozen"] = conformal_report(Tt, posthoc.rule_inputs(Tt), tau, edges, bwt, min_n)
        tb["conformal_by_alpha"] = {f"{a:g}": {"frozen": conformal_report(Tt, posthoc.rule_inputs(Tt, a), tau,
                                                                          edges, bwt, min_n)}
                                    for a in rq4_alphas}
        cal_path = predictions_path(cfg, det, name, "calib")
        if cal_path.exists():
            try:
                refit = PostHoc(fit_posthoc(
                    cfg, tables_from_records(cfg, load_records(cal_path), tmode, context_conf=ctx),
                    seed, tau=tau, mode=tmode))
                tb["conformal_recalibrated"] = conformal_report(Tt, refit.rule_inputs(Tt), tau, edges, bwt, min_n)
                for a in rq4_alphas:
                    tb["conformal_by_alpha"][f"{a:g}"]["recalibrated"] = conformal_report(
                        Tt, refit.rule_inputs(Tt, a), tau, edges, bwt, min_n)
            except ValueError as e:
                tb["recalibration_note"] = str(e)
        tb["occlusion"] = occlusion_analysis(Tt, tau, edges, bwt, occ_thr,
                                             int(ev.get("min_occluded_objects", 200)),
                                             int(ev.get("min_bucket_n", 30)), n_rank_boot=100,
                                             n_logit_boot=100, seed=seed)
        tb["comparison"] = transfer_summary(dev_ref, tb)
        out["modes"][tmode] = tb
        dt = tb["detection"]
        log.info(f"[{name}/test | {tmode}] {Tt.n_images} images, {Tt.n_gt} vehicles | recall@{tau:.2f} "
                 f"{_f(dt['recall_at_standard'])} | at floor {_f(dt['recall_at_floor'])}")
        if not tb["gate"]["usable"]:
            log.error(f"    GATE FAILED: {tb['gate']['message']}")
        by_alpha = tb["comparison"].get("conformal_coverage_by_alpha") or \
            {"headline": tb["comparison"]["conformal_coverage"]}
        for a, rows in by_alpha.items():
            for rule, c in rows.items():
                log.info(f"    {rule:<24} target {c['target']:.2f} | development {_f(c['development'])} | "
                         f"transfer frozen {_f(c['transfer_frozen'])} "
                         f"({_f(c['transfer_frozen_flags_per_image'], 2)} flags/image) | recalibrated "
                         f"{_f(c['transfer_recalibrated'])} "
                         f"({_f(c['transfer_recalibrated_flags_per_image'], 2)} flags/image, certifiable "
                         f"{c['transfer_recalibrated_certifiable']})")
        save_results(out, res / f"rq4_{name}.json")
        summary["gates"][f"detector_usable_on_{name}"] = {m: v["gate"] for m, v in out["modes"].items()}
        summary["headline"]["rq4"][name] = {m: v["comparison"] for m, v in out["modes"].items()}

    save_results(summary, res / "summary.json")
    log.info(f"Summary: {res / 'summary.json'}   Figures: {figs}")


if __name__ == "__main__":
    main()
