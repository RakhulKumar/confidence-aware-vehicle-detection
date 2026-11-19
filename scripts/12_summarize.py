#!/usr/bin/env python3
"""Collect the headline numbers of every detector and matching mode into one
readable file: outputs/<experiment>/digest.md (also printed).

  python scripts/12_summarize.py [--detectors yolo26 yolov8 rtdetr]

Reads only the JSON written by scripts/07 and 08. No GPU, instant. This is
the file to read first after a run, and the one to send back for review.
"""
from __future__ import annotations

from _common import setup

from confdet.pipeline import SCORE_LABELS
from confdet.utils.io_utils import load_results, matching_modes
from confdet.utils.paths import calibration_path, comparison_dir, experiment_dir, results_dir


def f(x, d=3):
    return "n/a" if x is None or x != x else f"{x:.{d}f}"


def ci(c, d=3):
    return "" if not c or c[0] is None or c[0] != c[0] else f" [{c[0]:.{d}f}, {c[1]:.{d}f}]"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


SEED_ROWS = [
    ("operating_threshold", "operating threshold"),
    ("ap50_class_agnostic", "AP50 (class-agnostic)"),
    ("recall_at_threshold", "recall at threshold"),
    ("false_alarms_per_image", "false alarms/image at threshold"),
    ("rq1_odds_ratio_per_0.1_occlusion", "RQ1 odds of a miss per +0.1 occlusion"),
    ("rq1_partial_spearman", "RQ1 partial Spearman"),
] + [(f"conformal_a{a}.{rule}.{q}", f"conformal {rule}, target {1 - float(a):.2f}: {lab}")
     for a in ("0.05", "0.1") for rule in ("global_confidence", "mondrian_confidence", "global_occlusion_aware")
     for q, lab in (("coverage", "coverage"), ("flags_per_image", "flags/image"))] + [
    ("rq2.mondrian_rank.delta_all", "RQ2 mondrian_rank vs confidence, all misses"),
    ("rq2.mondrian_rank.delta_occluded", "RQ2 mondrian_rank vs confidence, occluded misses"),
    ("rq2.occlusion_aware.delta_all", "RQ2 default score vs confidence, all misses"),
    ("rq2.occlusion_aware.delta_occluded", "RQ2 default score vs confidence, occluded misses"),
    ("dece", "D-ECE"), ("ma_ece", "MA-ECE"), ("ma_brier", "MA-Brier"), ("oce_max", "OCE"),
]


def seed_lines(summary: dict) -> list[str]:
    """Mean and SD across training seeds, per family (scripts/14)."""
    fams = summary.get("families", {})
    if not fams:
        return []
    out = [f"## Training-seed variability (matching: {summary.get('matching')})", "",
           "Mean, SD across seeds (n-1 denominator) and [min, max]. With three seeds the SD is itself "
           "rough: read it as the size of run-to-run noise.", ""]
    names = list(fams)
    rows = []
    for key, lab in SEED_ROWS:
        cells = []
        for fam in names:
            e = fams[fam]["metrics"].get(key)
            if not e or e.get("mean") is None:
                cells.append("n/a")
                continue
            d = 2 if "threshold" in key or "flags" in key or "false_alarms" in key else 4
            sd = "n/a" if e["sd"] is None else f"{e['sd']:.{d}f}"
            cells.append(f"{e['mean']:.{d}f} ± {sd} [{e['min']:.{d}f}, {e['max']:.{d}f}] (n={e['n']})")
        rows.append([lab] + cells)
    head = [f"{fam} ({', '.join(fams[fam]['members'])})" for fam in names]
    return out + table(["quantity"] + head, rows)


def brier_decomposition_lines(ranking: dict) -> list[str]:
    """MA-Brier = false-alarm term + matched-detection term + miss term, per detector, at its own
    threshold and at matched cost, next to OCE (which has no false-alarm term)."""
    rows = []
    for key, lab in (("at_own_threshold", "own threshold"), ("at_matched_cost", "matched cost")):
        for det, p in (ranking.get(key, {}).get("points") or {}).items():
            d = p.get("ma_brier_decomposition")
            if not d:
                continue
            rows.append([det, lab, f(p.get("threshold"), 2), f(p.get("fp_per_image"), 2), f(d["ma_brier"], 4)] +
                        [f"{f(d[t], 4)} ({f(d['share_' + t], 2)})" for t in ("false_alarm", "matched", "miss")] +
                        [f(p.get("oce_max"), 4)])
    if not rows:
        return []
    return ["#### MA-Brier decomposition", "",
            "Each term is its contribution to MA-Brier (share of MA-Brier in brackets); the three sum to "
            "MA-Brier. OCE averages over vehicles only, so it has no false-alarm term.", ""] + table(
        ["detector", "where", "threshold", "false alarms/image", "MA-Brier", "false-alarm term",
         "matched-detection term", "miss term", "OCE"], rows)


GUARANTEE = {
    "mondrian_confidence": "recall >= 1-a in every occlusion-context stratum (ours, headline)",
    "global_confidence": "recall >= 1-a, marginal (ours, reference)",
    "global_occlusion_aware": "recall >= 1-a, marginal; duplicate- and context-aware score (ours)",
    "global_gilg_iou_aware": "recall >= 1-a, marginal; Gilg IoU-aware score",
    "crc_fnr": "E[per-image FNR] <= a (CRC; Hulsman 2024)",
    "seqcrc_box_count_recall": "box-count recall loss <= a (SeqCRC; no spatial matching)",
    "seqcrc_box_count_threshold": "P(fewer boxes than objects) <= a (SeqCRC)",
    "pac": "P(object miss rate > a) <= delta (PAC; Li 2022)",
}


def baseline_lines(cfg: dict, dets: list[str], mode: str) -> list[str]:
    """Phase 1 (v0.6): literature baselines next to our rules (scripts/16)."""
    loaded = {}
    for det in dets:
        p = results_dir(cfg, det, mode) / "baselines.json"
        if p.exists():
            loaded[det] = load_results(p)
    if not loaded:
        return []
    out = ["### Baselines from the literature, next to our rules (v0.6 Phase 1)", "",
           "Every rule is fitted on the same calib images (the threshold half used by our conformal rules) and "
           "evaluated once on test. Coverage = object recall of the rule alone; flags = candidates the rule keeps "
           "below the operating threshold, per image; o_j = annotated occlusion, o_hat = occlusion context.", ""]
    for a in ("0.1", "0.05"):
        rows = []
        for det, r in loaded.items():
            blk = r["by_alpha"].get(a)
            if not blk:
                continue
            for name in GUARANTEE:
                v = blk["rules"].get(name)
                if not v:
                    continue
                if not v.get("achievable"):
                    rows.append([det, name, GUARANTEE[name], "not certifiable", "", "", "", ""])
                    continue
                top = v["by_annotated_occlusion"][-1]
                ctx = [s for s in v["by_occlusion_context"] if s["n"]]
                worst = min(ctx, key=lambda s: s["coverage"]) if ctx else None
                cert = v.get("calibration", {}).get("certifiable")
                rows.append([det, name + ("" if cert in (None, True) else " (NOT certifiable)"), GUARANTEE[name],
                             f(v["coverage"]) + ci(v["coverage_ci"]),
                             f(v["mean_image_fnr"]) + ci(v["mean_image_fnr_ci"]), f(v["flags_per_image"], 2),
                             f"{f(top['coverage'])} (n={top['n']})",
                             f"{f(worst['coverage'])} ({worst['lo']:.2f}-{worst['hi']:.2f})" if worst else "n/a"])
        if rows:
            out += [f"**Target 1-a = {1 - float(a):.2f}**" + ("  (primary, D6)" if a == "0.1" else ""), ""]
            out += table(["detector", "rule", "guarantee", "coverage", "per-image FNR", "flags/image",
                          "coverage, most occluded o_j bucket", "lowest coverage over o_hat strata"], rows)
    rows = []
    for det, r in loaded.items():
        g = r["gilg_vs_ours"]
        rec = g["recovery"]
        cells = []
        for k in ("gilg_iou_aware", "z_dup", "default_score"):
            e = rec.get(k, {})
            cells.append(f"{f(e.get('mean_recovery_delta_vs_confidence'), 4)}{ci(e.get('mean_recovery_delta_ci'), 4)}"
                         f" / occl. {f(e.get('mean_recovery_occluded_delta_vs_confidence'), 4)}")
        rows.append([det, f(g["spearman_gilg_vs_z_dup_weak_candidates"]), f(g["spearman_u_higher_vs_dup"])] + cells)
    out += ["#### IoU-aware calibration (Gilg et al.) vs our duplicate-aware scores", "",
            "Recovery of missed vehicles vs plain confidence at matched extra false alarms (RQ2 protocol), all "
            "misses / occluded misses. Gilg's score is fitted on the calib score half (no fit-role predictions exist).",
            ""] + table(["detector", "Spearman: Gilg vs z_dup (weak candidates)",
                         "Spearman: Gilg's u vs our dup feature", "Gilg IoU-aware", "our z_dup",
                         "our default (duplicate- and context-aware)"], rows)
    rows = []
    for det, r in loaded.items():
        c, lr = r["calibration_metrics"], r["lrp"]
        a, b = lr["operating_threshold_max_f1"], lr["lrp_optimal_threshold_from_calib"]
        rows.append([det, f(c["laece0"], 4) + ci(c["laece0_ci"], 4), f(c["laace0"], 4) + ci(c["laace0_ci"], 4),
                     f(c["dece"], 4), f"{f(a['threshold'], 2)}: {f(a['lrp'], 4)}{ci(a['lrp_ci'], 4)}",
                     f"{f(b['threshold'], 2)}: {f(b['lrp'], 4)}{ci(b['lrp_ci'], 4)}",
                     f"{f(b['recall'])} / {f(b['fp_per_image'], 2)}"])
    out += ["#### Localisation-aware calibration and LRP (Kuzucu et al. 2024)", "",
            "LaECE_0 / LaACE_0: 25 bins, target = IoU with the matched vehicle (any IoU > 0), at the operating "
            "threshold. LRP at tau = 0.5; the LRP-optimal threshold is chosen on calib.", ""] + table(
        ["detector", "LaECE_0", "LaACE_0", "D-ECE", "LRP at F1-optimal threshold", "LRP at LRP-optimal threshold",
         "recall / false alarms per image at LRP-optimal"], rows)
    return out


def display_name(key: str, score_features: str) -> str:
    """Reader-facing name of a policy / conformal rule. The fitted score keeps the result key
    "occlusion_aware" whatever its feature set; its label says what it actually uses."""
    name = SCORE_LABELS.get(score_features, score_features)
    if key == "occlusion_aware":
        return name
    if key == "global_occlusion_aware":
        return f"global, {name}"
    return key


def rq1_size_lines(loaded: dict) -> list[str]:
    """RQ1 by size tertile, and occlusion bucket x size tertile."""
    have = {det: r["rq1"] for det, r in loaded.items() if r.get("rq1", {}).get("by_size_tertile")}
    if not have:
        return []
    out = ["#### RQ1 by object size (tertiles of box area relative to the image)", ""]
    rows = []
    for det, q in have.items():
        lo, hi = q["size_tertile_edges_rel_area"]
        cells = []
        for row in q["by_size_tertile"]:
            cells.append(f"{f(row['miss_rate'])}{ci(row['miss_rate_ci'])} (n={row['n']})")
        rows.append([det, f"{lo:.2e} / {hi:.2e}"] + cells)
    out += table(["detector", "tertile edges (relative area)", "small", "medium", "large"], rows)
    out += ["Miss rate by occlusion bucket within each size tertile (n in brackets; `*` = fewer than "
            "`min_bucket_n`):", ""]
    rows = []
    any_q = next(iter(have.values()))
    buckets = sorted({(c["bucket"], c["lo"], c["hi"]) for c in any_q["by_occlusion_and_size"]})
    for det, q in have.items():
        cell = {(c["bucket"], c["size"]): c for c in q["by_occlusion_and_size"]}
        for size in ("small", "medium", "large"):
            rows.append([det, size] + [
                f"{f(cell[(b, size)]['miss_rate'])} ({cell[(b, size)]['n']})"
                + ("" if cell[(b, size)]["reliable"] else "*") for b, _, _ in buckets])
    out += table(["detector", "size"] + [f"{lo:.1f}-{hi:.1f}" for _, lo, hi in buckets], rows)
    return out


def ablation_lines(cfg: dict, dets: list[str], mode: str) -> list[str]:
    """Ablation of the occlusion-aware score (scripts/13), if it was run for this mode."""
    rows = []
    for det in dets:
        d = calibration_path(cfg, det, mode).parent
        if not (d / "ablation_selection.json").exists() or not (d / "ablation_test.json").exists():
            continue
        sel, res = load_results(d / "ablation_selection.json"), load_results(d / "ablation_test.json")
        for name, cv in sel["table"].items():
            t = res["sets"].get(name, {})
            rows.append([det, name + (" **(chosen on calib)**" if name == sel["chosen"] else ""),
                         f(cv.get("cv_mean_recovery"), 4), f(cv.get("cv_mean_recovery_occluded"), 4),
                         f(t.get("mean_recovery_over_budget_range")),
                         f(t.get("mean_recovery_delta_vs_confidence"), 4) + ci(t.get("mean_recovery_delta_ci"), 4),
                         f(t.get("mean_recovery_occluded_delta_vs_confidence"), 4)
                         + ci(t.get("mean_recovery_occluded_delta_ci"), 4)])
    if not rows:
        return []
    return ["### RQ2 ablation: which features of the occlusion-aware score carry the gain", "",
            "Chosen by 5-fold cross-validation over calib images (criterion fixed in advance: mean recovery of "
            "all misses over the budget range); every set then evaluated once on test against plain "
            "confidence. The configured default calibrator is unchanged.", ""] + table(
        ["detector", "feature set", "calib CV: mean recovery", "calib CV: occluded", "test: mean recovery",
         "test: difference from confidence, all misses", "test: difference, occluded misses"], rows)


def cert(v):
    return {True: "yes", False: "NO"}.get(v, "n/a")


def cov_cell(cov, flags, certifiable):
    return f"{f(cov)} ({f(flags, 2)}; {cert(certifiable)})"


def rq4_rows(loaded: dict) -> list[list[str]]:
    """RQ4 conformal rows, one per detector x dataset x mode x alpha x rule,
    every coverage with its flags per image and certifiability."""
    rows = []
    for det, r in loaded.items():
        for ds, modes in r["summary"]["headline"].get("rq4", {}).items():
            for tmode, cmp in modes.items():
                m = cmp["metrics"]
                recall = f"{f(m['recall_at_standard']['development'])} -> {f(m['recall_at_standard']['transfer'])}"
                by_alpha = cmp.get("conformal_coverage_by_alpha") or {"": cmp["conformal_coverage"]}
                for _, cov in sorted(by_alpha.items(), key=lambda kv: float(kv[0] or 0)):
                    for rule, c in cov.items():
                        rows.append([det, ds, tmode, recall, rule, f(c["target"], 2),
                                     cov_cell(c["development"], c.get("development_flags_per_image"),
                                              c.get("development_certifiable")),
                                     cov_cell(c["transfer_frozen"], c.get("transfer_frozen_flags_per_image"),
                                              c.get("transfer_frozen_certifiable")),
                                     cov_cell(c["transfer_recalibrated"],
                                              c.get("transfer_recalibrated_flags_per_image"),
                                              c.get("transfer_recalibrated_certifiable"))])
    return rows


def main():
    args, cfg, log = setup("summarize", __doc__,
                           extra=lambda p: p.add_argument("--detectors", nargs="+", default=None))
    dets = args.detectors or list(cfg["detectors"]["models"])
    lines = [f"# Results digest: experiment `{cfg['experiment']['name']}`", "",
             "Intervals are 95% image-level bootstrap intervals. Read the gates first.", ""]
    for mode in matching_modes(cfg):
        loaded = {}
        for det in dets:
            res = results_dir(cfg, det, mode)
            if (res / "summary.json").exists():
                loaded[det] = {k: load_results(res / f"{k}.json") for k in
                               ("summary", "rq1", "rq2", "rq3", "conformal") if (res / f"{k}.json").exists()}
                if (res / "rq5.json").exists():
                    loaded[det]["rq5"] = load_results(res / "rq5.json")
                cal = calibration_path(cfg, det, mode)
                loaded[det]["score_features"] = (load_results(cal)["meta"].get("score_features", "occlusion_aware")
                                                 if cal.exists() else "occlusion_aware")
        if not loaded:
            continue
        lines += [f"## Matching: {mode}" + ("  (primary)" if mode == matching_modes(cfg)[0] else ""), ""]

        lines += ["### Gates", ""]
        for det, r in loaded.items():
            for gate, v in r["summary"]["gates"].items():
                items = v.items() if "usable" not in v and "adequate" not in v else [("", v)]
                for sub, g in items:
                    ok = g.get("usable", g.get("adequate"))
                    lines.append(f"- {det} / {gate}{(' / ' + sub) if sub else ''}: "
                                 f"{'pass' if ok else 'FAIL -- ' + str(g.get('message'))}")
        lines.append("")

        lines += ["### Detection at each detector's operating threshold", ""]
        lines += table(["detector", "threshold", "AP50", "recall", "precision", "false alarms/image", "recall at floor"],
                       [[det, f(r["summary"]["standard_threshold"], 2)] +
                        [f(r["summary"]["headline"]["detection"][k]) for k in
                         ("ap50", "recall_at_standard", "precision_at_standard")] +
                        [f(r["summary"]["headline"]["detection"]["fp_per_image_at_standard"], 2),
                         f(r["summary"]["headline"]["detection"]["recall_at_floor"])] for det, r in loaded.items()])

        lines += ["### RQ1: miss rate by annotated occlusion severity", ""]
        any_r = next(iter(loaded.values()))["rq1"]
        heads = [f"{b['lo']:.1f}-{b['hi']:.1f} (n={b['n']})" for b in any_r["buckets"]]
        rows = []
        for det, r in loaded.items():
            q = r["rq1"]
            m = q["logistic_miss_model"]
            rows.append([det] + [f(b.get("miss_rate")) + ("*" if not b["reliable"] else "") for b in q["buckets"]] +
                        [f(m.get("odds_ratio_per_0.1_occlusion")) + ci(m.get("odds_ratio_ci")),
                         f(q["partial_spearman_given_size_and_class"]) + ci(q["partial_spearman_ci"]),
                         f(q["spearman_occlusion_vs_deployable_context"], 2)])
        lines += table(["detector"] + heads + ["odds of a miss per +0.1 (size, class controlled)",
                                               "partial Spearman", "context vs annotated occlusion"], rows)
        lines += ["`*` marks a bucket with fewer objects than `min_bucket_n`; do not read it.", ""]
        lines += rq1_size_lines(loaded)

        lines += ["### Conformal layer", ""]
        rows = []
        for det, r in loaded.items():
            rep = r["conformal"]["report"]
            for rule, v in rep.items():
                sysb = v.get("system_by_occlusion_bucket", v["by_occlusion_bucket"])[-1]
                rows.append([det, display_name(rule, r["score_features"]), f(v.get("target_coverage"), 2),
                             f(v["coverage"]) + ci(v["coverage_ci"]), f(v.get("system_recall", v["coverage"])),
                             f(sysb["coverage"]), f(v.get("flagged_per_image"), 2),
                             {True: "yes", False: "NO", None: ""}[v.get("all_groups_achievable")]])
        lines += table(["detector", "rule", "target", "recall of rule alone", "recall of system",
                        "system recall, most occluded bucket", "flags/image", "certifiable"], rows)

        lines += ["### RQ2: recovery of missed vehicles at matched extra false alarms", ""]
        rows = []
        for det, r in loaded.items():
            q = r["rq2"]
            for pol, e in q.get("policies", {}).items():
                rows.append([det, display_name(pol, r["score_features"]), q["n_missed_at_standard"], f(q["recoverable_ceiling"]),
                             f(e["mean_recovery_over_budget_range"]) + ci(e["mean_recovery_ci"]),
                             (f(e.get("mean_recovery_delta_vs_confidence"), 4) + ci(e.get("mean_recovery_delta_ci"), 4))
                             if "mean_recovery_delta_vs_confidence" in e else "baseline",
                             (f(e.get("mean_recovery_occluded_delta_vs_confidence"), 4) +
                              ci(e.get("mean_recovery_occluded_delta_ci"), 4))
                             if "mean_recovery_occluded_delta_vs_confidence" in e else "baseline"])
        lines += table(["detector", "policy", "misses", "recoverable ceiling", "mean recovery",
                        "difference from baseline, all misses", "difference from baseline, occluded misses"], rows)

        lines += ablation_lines(cfg, list(loaded), mode)
        lines += baseline_lines(cfg, list(loaded), mode)

        lines += ["### RQ3: calibration metrics at the operating threshold", ""]
        rows = []
        for det, r in loaded.items():
            p = r["rq3"]["at_standard_threshold"]
            pref = r["rq3"]["threshold_each_metric_prefers"]
            dec = r["rq3"]["gap_decomposition_by_occlusion"]
            occ_share_obj = sum(b["share_of_objects"] or 0 for b in dec[1:])
            occ_share_gap = sum(b["share_of_gap"] or 0 for b in dec[1:])
            rows.append([det, f(p["dece"]) + ci(p["dece_ci"]), f(p["ma_ece"]) + ci(p["ma_ece_ci"]), f(p["miss_mass"]),
                         f(p["oce_max"]),
                         f"{pref['dece']['threshold']:g} (recall {f(pref['dece']['recall_there'], 2)})" if "dece" in pref else "",
                         f"{f(occ_share_obj, 2)} / {f(occ_share_gap, 2)}"])
        lines += table(["detector", "D-ECE", "MA-ECE", "miss mass", "OCE", "threshold D-ECE alone would choose",
                        "occluded vehicles: share of objects / share of D-ECE blind spot"], rows)

        if any("rq5" in r for r in loaded.values()):
            lines += ["### RQ5: why vehicles are missed (native classes, whatever the mode)", ""]
            rows = []
            for det, r in loaded.items():
                q = r.get("rq5")
                if not q:
                    continue
                v, c = q.get("visibility_miss_model", {}), q.get("confusion_miss_model", {})
                rows.append([det, q["n_missed"]] + [f(q["cause_shares"].get(k), 2) for k in
                                                    ("confused", "suppressed", "unproposed", "merged")] +
                            [f(v.get("odds_ratio_per_0.1_occlusion")) + ci(v.get("occlusion_ci")),
                             f(c.get("odds_ratio_per_sd_rarity")) + ci(c.get("rarity_ci")),
                             f(c.get("odds_ratio_per_0.1_occlusion")) + ci(c.get("occlusion_ci"))])
            lines += table(["detector", "misses", "confused", "suppressed", "unproposed", "merged",
                            "visibility misses: odds per +0.1 occlusion", "confusion misses: odds per +1 SD rarity",
                            "confusion misses: odds per +0.1 occlusion"], rows)

        rows = rq4_rows(loaded)
        if rows:
            lines += ["### RQ4: transfer with the detector frozen (recall-side quantities only)", ""]
            lines += ["Each coverage is followed by (flags per image; certifiable). A recalibrated coverage "
                      "that is not certifiable is the best the rule could do, not a guarantee.", ""]
            lines += table(["detector", "dataset", "matching", "recall, development -> transfer", "rule", "target",
                            "coverage on development", "coverage on transfer, frozen",
                            "coverage on transfer, recalibrated"], rows)

        rk = comparison_dir(cfg, mode) / "ranking.json"
        if rk.exists():
            k = load_results(rk)
            lines += ["### Detector ranking (best first)", ""]
            rows = []
            for key, label in (("at_own_threshold", "each at its own threshold"),
                               ("at_matched_cost", "at matched false alarms per image")):
                for metric, order in k.get(key, {}).get("rankings", {}).items():
                    rows.append([label, metric, " < ".join(order)])
            lines += table(["comparison", "metric", "ranking"], rows)
            lines += ["Point-estimate orders above can differ by noise. Below, a metric *separates* a pair only "
                      "if its 95% paired-difference interval excludes zero; a *flip* needs both D-ECE and the "
                      "other metric to separate the pair, in opposite directions.", ""]
            for key, label in (("at_own_threshold", "each at its own threshold"),
                               ("at_matched_cost", "at matched false alarms per image")):
                block = k.get(key, {})
                if "pairwise" not in block:
                    continue
                rows = []
                for row in block["pairwise"]:
                    cells = []
                    for metric in ("dece", "ma_ece", "ma_brier", "oce_max"):
                        e = row[metric]
                        verdict = (f"{e['better_if_separated']} better" if e.get("separates")
                                   else "cannot separate")
                        cells.append(f"{verdict}{ci(e['difference_a_minus_b_ci'], 4)}")
                    flips = [m for m, v in row["flips_vs_dece"].items() if v]
                    rows.append([f"{row['a']} vs {row['b']}"] + cells + [", ".join(flips) or "none"])
                lines += [f"**{label}**" + (f" (budget {f(block.get('fp_per_image_budget'), 2)} false alarms/image)"
                                            if "fp_per_image_budget" in block else ""), ""]
                lines += table(["pair (difference = first minus second)", "D-ECE", "MA-ECE", "MA-Brier", "OCE",
                                "flips vs D-ECE"], rows)
                for a, b in block.get("pairs_dece_cannot_separate", []):
                    lines.append(f"- D-ECE cannot separate {a} and {b} ({label}).")
                lines.append("")
            lines += [f"Ranking flip relative to D-ECE (both intervals exclude zero, opposite signs) survives "
                      f"matched cost: **{k.get('ranking_flip_survives_matched_cost')}**", ""]
            lines += brier_decomposition_lines(k)

    seeds = experiment_dir(cfg) / "seeds" / "seed_summary.json"
    if seeds.exists():
        lines += seed_lines(load_results(seeds))

    out = experiment_dir(cfg) / "digest.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    log.info(f"Saved {out}")


if __name__ == "__main__":
    main()
