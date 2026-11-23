#!/usr/bin/env python3
"""v0.6 interim report: NMS pilot, evidential pilot, UA-DETRAC, queue status -> outputs/v06_interim_report.md.

  python scripts/21_interim_report.py [--queue <paths_local.yaml: queue> | --queue none]
         [--experiment bmd_dev] [--evid yolo26_evid_pilot_b[,more]] [--out FILE]

  Defaults reproduce the bmd_dev report. For the final models:
    --experiment bmd_v06 --evid yolo26_evid,yolo26_evid_s1,yolo26_evid_s2 --queue none
  reads the same files under outputs/bmd_v06/ and writes outputs/bmd_v06/interim_report.md.

Reads only result files (no GPU, seconds). Every number is copied from (bmd_dev = --experiment):
  outputs/bmd_dev/yolo26/class_agnostic/results/nms_pilot.json
  outputs/bmd_dev/yolo26_evid_pilot_b/class_agnostic/results/evidential_pilot.json
  outputs/bmd_dev/<det>/class_agnostic/results/ua_detrac_transfer.json
  outputs/bmd_dev/<det>/class_agnostic/results/ua_detrac_certifiability.json   (scripts/20b; table skipped if absent)
  outputs/bmd_dev/comparison/class_agnostic/ua_detrac_proxy_validation.json
A section whose file is missing says so instead of inventing numbers.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from confdet.utils.io_utils import site_paths  # noqa: E402

OUT = REPO / "outputs"
R = OUT / "bmd_dev"


def f(x, d=3):
    return "n/a" if x is None or x != x else f"{x:.{d}f}"


def ci(c, d=3):
    return "" if not c or c[0] is None or c[0] != c[0] else f" [{c[0]:.{d}f}, {c[1]:.{d}f}]"


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""]


def nms_section():
    d = load(R / "yolo26" / "class_agnostic" / "results" / "nms_pilot.json")
    if not d:
        return ["## a) NMS pilot", "", "_nms_pilot.json not found: the job has not finished._", ""]
    m = d["miss_mechanism_test"]
    out = ["## a) NMS pilot (yolo26 seed 0, BMD-45 test, class-agnostic)", "",
           f"Pre-NMS candidates: {f(d['prenms_candidates_per_image'], 0)} per image (cap {d['prenms_cap_per_image']}). "
           f"Operating threshold {f(d['tau'], 2)}; {m['n_missed']} missed vehicles.", "",
           "**Why each missed vehicle is missed** (share of misses):", ""]
    rows = [["all", m["n_missed"]] + [f(m["shares"][k]) for k in m["shares"]]]
    for b in m["by_annotated_occlusion"]:
        rows.append([f"o_j {b['lo']:.1f}-{b['hi']:.1f}", b["n_missed"]] + [f(b["shares"][k]) for k in m["shares"]])
    out += table(["annotated occlusion", "misses"] + list(m["shares"]), rows)
    out += ["**Suppression variants at matched false alarms** (recall; recall by annotated occlusion bucket):", ""]
    rows = []
    for name, v in d["variants"].items():
        for r in v["at_matched_false_alarms"]:
            rows.append([name, f(v["ap50"], 4), f(r["fp_per_image"], 2), f(r["recall"], 4),
                         " / ".join(f(x) for x in r["recall_by_annotated_occlusion"])])
    out += table(["variant", "AP50", "false alarms/image", "recall", "recall by o_j bucket"], rows)
    return out


def evid_section(det="yolo26_evid_pilot_b"):
    d = load(R / det / "class_agnostic" / "results" / "evidential_pilot.json")
    if not d:
        return [f"## b) Evidential pilot ({det})", "", "_evidential_pilot.json not found: the job has not finished._", ""]
    a = d["associations"]
    s = d["uncertainty_split"]
    gap = d.get("ap50_gap_to_yolo26_family_mean")
    out = [f"## b) Evidential pilot ({det}: YOLO26, Beta evidence head, current fit set)", "",
           f"AP50 class-agnostic {f(d['ap50_class_agnostic'], 4)} vs yolo26 family mean "
           f"{f(d['yolo26_family_ap50_class_agnostic_mean'], 4)} (gap {f(gap, 4)}); class-aware {f(d['ap50_class_aware'], 4)}. "
           f"Threshold {f(d['threshold_max_f1_calib'], 2)}: recall {f(d['recall_at_threshold'])}, "
           f"{f(d['fp_per_image'], 2)} false alarms/image. {d['note']}.", "",
           "Per vehicle (its matched candidate), Spearman and partial Spearman controlling box size, image-bootstrap 95% CI:", ""]
    rows = [[k.replace("_vs_", " vs "), f(v["spearman"]) + ci(v["spearman_ci"]), f(v["partial_spearman_size"]) + ci(v["partial_ci"])]
            for k, v in a.items()]
    out += table(["association", "Spearman", "partial (size)"], rows)
    out += [f"Uncertainty split: Spearman(aleatoric, epistemic) = {f(s['spearman_aleatoric_vs_epistemic'])}; "
            f"rho(epistemic, rarity) - rho(aleatoric, rarity) = {f(s['rarity_diff_epi_minus_alea'])}{ci(s['rarity_diff_ci'])}; "
            f"rho(aleatoric, o_j) - rho(epistemic, o_j) = {f(s['occlusion_diff_alea_minus_epi'])}{ci(s['occlusion_diff_ci'])}.", ""]
    if not det.startswith("yolo26_evid_pilot"):
        return out + ["These are the final v0.6 evidential models (one of three seeds); the pilot's gate and the RT-DETR "
                      "feasibility note do not apply to them.", ""]
    out += [f"Gate for evidential x3 (decided 14 Nov): AP50 within 0.02 of baseline -> "
            f"**{'met' if gap is not None and gap >= -0.02 else 'NOT met'}**; usable uncertainty split -> for review "
            f"(the two parts must not be near-collinear and must differ in what they follow).", "",
            "RT-DETR feasibility: its class scores come from the decoder's score head trained with a varifocal/focal "
            "loss inside DETRLoss; the same Beta head means doubling that head's outputs and replacing the varifocal "
            "term, with the Hungarian matcher kept on p = alpha/S. Feasible, more invasive than YOLO26; not attempted "
            "in this pilot.", ""]
    return out


def evid_rarity_section(det="yolo26_evid_pilot_b"):
    d = load(R / det / "class_agnostic" / "results" / "evidential_rarity_diag_calib.json")
    out = [f"### evidential: rarity diagnostic (calib, {det})", ""]
    if not d:
        return out + ["_evidential_rarity_diag_calib.json not found (scripts/22_evid_rarity_diag.py)._", ""]
    out += [f"Calib role only, {d['n_vehicles_with_box']} vehicles with a box, threshold {f(d['threshold'], 2)}. "
            "Hypothesis: evidence is per fine class, so the common, confusable car subtypes split it. Spearman with "
            "image-bootstrap 95% CI.", ""]
    rows = []
    for key, label in (("a_epistemic_all", "(a) epistemic, all vehicles"), ("a_aleatoric_all", "(a) aleatoric, all vehicles"),
                       ("b_epistemic_confusion_misses", "(b) epistemic, confusion misses"),
                       ("b_epistemic_visibility_misses", "(b) epistemic, visibility misses")):
        v = d[key]
        if "note" in v:
            rows.append([label, v["n"], v["note"], ""])
            continue
        rows.append([label, v["n"], f(v["fine_rarity"]["spearman"]) + ci(v["fine_rarity"]["spearman_ci"]),
                     f(v["coarse_rarity"]["spearman"]) + ci(v["coarse_rarity"]["spearman_ci"])])
    out += table(["group", "n", "vs fine-class rarity", "vs coarse-class rarity"], rows)
    c = d["c_car_subtypes_vs_rare"]
    out += [f"(c) mean epistemic, car subtypes ({', '.join(c['car_subtypes'])}) {f(c['mean_epistemic_car_subtypes'], 5)} vs "
            f"rare classes (fit share < {c['rare_share_below']:.0%}: {', '.join(c['rare_classes'])}) {f(c['mean_epistemic_rare'], 5)}; "
            f"difference {f(c['difference'], 5)}{ci(c['difference_ci'], 5)}.", ""]
    out += table(["class", "fit share", "n", "mean epistemic", "mean aleatoric"],
                 [[r["class"], f(r["fit_share"]), r["n"], f(r["mean_epistemic"], 5), f(r["mean_aleatoric"], 4)]
                  for r in d["c_per_class"]])
    return out


def detrac_section():
    pv = load(R / "comparison" / "class_agnostic" / "ua_detrac_proxy_validation.json")
    out = ["## c) UA-DETRAC (new cameras; current models, zero-shot)", ""]
    if not pv:
        return out + ["_ua_detrac_proxy_validation.json not found: the job has not finished._", ""]
    p1, p2 = pv["box_overlap_proxy_vs_occ_vehicle"], pv["box_overlap_proxy_vs_occ_naive"]
    out += [f"**Proxy validation** ({pv['n_vehicles']} test vehicles, sequence-bootstrap CI): box-overlap proxy vs annotated "
            f"occlusion = {f(p1[0])}{ci(p1[1])} (confirmed reading), {f(p2[0])}{ci(p2[1])} (naive reading). "
            f"{f(pv['share_vehicles_occluded_by_vehicle_ge_0.2'])} of vehicles have annotated vehicle-occlusion >= 0.2; "
            f"{f(pv['share_occluded_by_nonvehicle_gt_0'])} have some non-vehicle occlusion; "
            f"{f(pv['share_truncated_gt_0'])} are truncated.", ""]
    rows = [[det, f(v["o_hat_vs_occ_vehicle"][0]) + ci(v["o_hat_vs_occ_vehicle"][1]),
             f(v["o_hat_vs_occ_naive"][0]) + ci(v["o_hat_vs_occ_naive"][1])] for det, v in pv["o_hat_per_detector"].items()]
    out += table(["detector", "o_hat vs annotated (confirmed)", "o_hat vs annotated (naive)"], rows)
    for det in pv["o_hat_per_detector"]:
        d = load(R / det / "class_agnostic" / "results" / "ua_detrac_transfer.json")
        if not d:
            continue
        out += [f"**{det}**: {d['n_images']} test frames, {d['n_sequences']} sequences, {d['n_vehicles']} vehicles. Recall at the "
                f"frozen threshold {f(d['frozen_threshold'], 2)}: {f(d['recall_at_frozen_threshold'])}"
                f"{ci(d['recall_ci_sequence_bootstrap'])} ({f(d['fp_per_image_at_frozen_threshold'], 2)} false alarms/image); "
                f"at the confidence floor {f(d['recall_at_floor'])}.", ""]
        rows = []
        for a in ("0.1", "0.05"):
            for k, v in d["by_alpha"].get(a, {}).items():
                top = v["by_annotated_occlusion"][-1]
                ctxs = [s for s in v["by_occlusion_context"] if s["n"]]
                worst = min(ctxs, key=lambda s: s["coverage"]) if ctxs else None
                rows.append([f"{1 - float(a):.2f}", k, f(v["coverage"]) + ci(v["coverage_ci"]), f(v["flags_per_image"], 2),
                             "yes" if v["certifiable"] else "NO",
                             " / ".join(f(b["coverage"]) for b in v["by_annotated_occlusion"]),
                             f"{f(worst['coverage'])}" if worst else "n/a", f"{top['n']}"])
        out += table(["target", "rule (setting:rule)", "coverage", "flags/image", "certifiable",
                      "coverage by annotated occlusion bucket", "lowest o_hat stratum", "n in top bucket"], rows)
        out += certifiability_table(det)
    return out


def certifiability_table(det):
    """Per o_hat group of the recalibrated Mondrian rules: why certifiable or not (scripts/20b)."""
    c = load(R / det / "class_agnostic" / "results" / "ua_detrac_certifiability.json")
    if not c:
        return []
    rows = []
    for a in ("0.1", "0.05"):
        for k, groups in c["by_alpha"].get(a, {}).items():
            if not k.startswith("recalibrated:"):
                continue
            for g in groups:
                rows.append([f"{1 - float(a):.2f}", k, f"{g['o_hat_lo']:.2f}-{g['o_hat_hi']:.2f}", g["n_calib"],
                             f"{g['n_never_proposed']} ({f(g['share_never_proposed'])})", f(g["min_certifiable_alpha"]),
                             f"{g['target_alpha']:g}", "yes" if g["certifiable"] else "NO"])
    return [f"Why the recalibrated Mondrian rules are not certifiable ({det}; UA-DETRAC calib, per o_hat group; a group is "
            "certifiable only if its smallest certifiable miss rate (never proposed + 1) / (n + 1) is at most the target alpha):", ""] + \
        table(["target", "rule", "o_hat group", "calib n", "never proposed", "min certifiable alpha", "target alpha", "certifiable"], rows)


def queue_section(qpath):
    try:
        st = subprocess.run([sys.executable, str(REPO / "scripts" / "run_queue.py"), "--queue", qpath, "--status"],
                            capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa: BLE001
        st = f"(status unavailable: {e})"
    return ["## Queue status", "", "```", st.strip(), "```", ""]


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    global R
    exp = _arg("--experiment", "bmd_dev")
    R = OUT / exp
    evid = [e for e in _arg("--evid", "yolo26_evid_pilot_b").split(",") if e]
    qpath = _arg("--queue") or site_paths()["queue"]
    title = "# v0.6 interim report" + ("" if exp == "bmd_dev" else f": experiment `{exp}`")
    lines = [f"{title} ({datetime.now():%Y-%m-%d %H:%M})", "",
             "Written automatically by scripts/21_interim_report.py when jobs a) to c) finished; numbers are copied from "
             "the result files named in the script. Interpretation is left to review.", ""]
    lines += nms_section()
    for det in evid:
        lines += evid_section(det) + evid_rarity_section(det)
    lines += detrac_section() + ([] if qpath.lower() == "none" else queue_section(qpath))
    default_out = OUT / "v06_interim_report.md" if exp == "bmd_dev" else R / "interim_report.md"
    out = Path(_arg("--out")) if "--out" in sys.argv else default_out
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
