#!/usr/bin/env python3
"""End-to-end self-test of the whole pipeline. No GPU, no Ultralytics, no
real data; takes about a minute.

  python scripts/00_selftest.py [--keep]

It builds two synthetic datasets with different class taxonomies in a temp
folder, then runs the REAL scripts 02 -> 08 on them with a simulated detector
whose behaviour is known by construction (evaluation/simulate.py), and
checks that the pipeline's answers match what was built in:

  * no leakage group straddles two roles;
  * misses rise with occlusion, and the analysis finds exactly that;
  * conformal thresholds deliver the recall they promise on held-out images;
  * the duplicate- and context-aware score beats plain threshold-lowering at equal cost
    (the simulated world is built so that it should);
  * MA-ECE = (1 - phi) * D-ECE + phi holds exactly;
  * the cross-dataset evaluation maps classes by NAME through the common
    taxonomy (the first real run compared raw class indices and found nothing);
  * a cautious detector that stays silent is ranked below a detector that
    finds the vehicles once misses count.

Run this after installing and after any code change, before spending GPU
hours. It exercises everything except the Ultralytics train/predict calls.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from confdet.evaluation.simulate import SimConfig, simulate_scene  # noqa: E402

DEV_NAMES = ["car", "rickshaw", "bike", "bus", "tempo"]            # detector taxonomy
OTHER_NAMES = ["Coach", "Scooter", "Sedan", "Cart", "Auto", "Van"]  # different names, different order
COMMON = ["car", "three_wheeler", "two_wheeler", "bus", "van"]
DEV_MAP = {"car": "car", "rickshaw": "three_wheeler", "bike": "two_wheeler", "bus": "bus", "tempo": "van"}
OTHER_MAP = {"Coach": "bus", "Scooter": "two_wheeler", "Sedan": "car", "Cart": None, "Auto": "three_wheeler",
             "Van": "van"}
# detector class id for each OTHER class id (what a real detector would answer): -1 = nothing equivalent
OTHER_TO_DEV = [3, 2, 0, -1, 1, 4]


def write_dataset(root: Path, names: list[str], n_images: int, seed: int, video_share: float,
                  class_ids: list[int] | None = None):
    from PIL import Image

    rng = np.random.default_rng(seed)
    cfg = SimConfig(width=640, height=360)
    for i in range(n_images):
        split = ("train", "valid", "test")[i % 3]
        xyxy, cls = simulate_scene(rng, cfg)
        if class_ids is not None:                         # express the scene in this dataset's own class ids
            cls = np.array([class_ids[c] for c in cls], dtype=int)
            if i % 9 == 0 and len(cls):
                cls[0] = names.index("Cart")              # a class the detector has no counterpart for
        stem = (f"cam_a_mp4-{i}_jpg.rf.{seed:02d}{i:06d}" if rng.random() < video_share
                else f"web-{i}_jpg.rf.{seed:02d}{i:06d}")
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (cfg.width, cfg.height), (110, 115, 120)).save(root / split / "images" / f"{stem}.jpg",
                                                                        quality=30)
        lines = [f"{c} {(b[0] + b[2]) / 2 / cfg.width:.6f} {(b[1] + b[3]) / 2 / cfg.height:.6f} "
                 f"{(b[2] - b[0]) / cfg.width:.6f} {(b[3] - b[1]) / cfg.height:.6f}" for b, c in zip(xyxy, cls)]
        (root / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    (root / "data.yaml").write_text(
        f"train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: {len(names)}\nnames: {names}\n")


def make_config(work: Path) -> Path:
    base = yaml.safe_load((REPO / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    base["project"].update({"output_dir": "outputs", "processed_dir": "data/processed"})
    base["experiment"] = {"name": "selftest", "development": "sim_dense", "transfer": ["sim_other"]}
    base["datasets"] = {
        "sim_dense": {"root": "raw/sim_dense", "label_format": "bbox", "video_segment_frames": 20,
                      "roles": [{"from": ["train", "val", "test"],
                                 "to": {"fit": 0.2, "val": 0.1, "calib": 0.35, "test": 0.35}}],
                      "class_map": DEV_MAP},
        "sim_other": {"root": "raw/sim_other", "label_format": "bbox",
                      "roles": [{"from": ["train", "val", "test"], "to": {"calib": 0.4, "test": 0.6}}],
                      "class_map": OTHER_MAP},
    }
    foreign = {"sim_other": OTHER_TO_DEV}
    base["detectors"] = {"active": "sim_a", "abort_on_vram_overflow": True, "models": {
        "sim_a": {"family": "simulated", "simulation": {"foreign_class_ids": foreign}},
        # stays silent on a third of the vehicles, impeccable on the rest
        "sim_cautious": {"family": "simulated", "simulation": {
            "foreign_class_ids": foreign, "p_unproposed_base": 0.33, "background_fp_rate": 0.2,
            "duplicate_rate": 0.05, "rare_confusion": 0.0, "logit_noise": 0.5}},
        # training-seed replicates of sim_a (v0.5): identical except train_args.seed
        "sim_a_s1": {"family": "simulated", "simulation": {"foreign_class_ids": foreign}, "train_args": {"seed": 1}},
        "sim_a_s2": {"family": "simulated", "simulation": {"foreign_class_ids": foreign}, "train_args": {"seed": 2}},
    }}
    base["posthoc"]["alpha"] = 0.1
    base["posthoc"]["alpha_grid"] = [0.1, 0.2, 0.3]
    base["evaluation"]["matching_modes"] = ["class_agnostic", "native"]
    base["evaluation"]["standard_operating_threshold"] = "max_f1"
    base["evaluation"]["bootstrap_replicates"] = 120
    base["evaluation"]["min_occluded_objects"] = 150
    base["evaluation"]["common_classes"] = COMMON
    path = work / "selftest.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return path


def run(script: str, work: Path, config: Path, *extra: str) -> None:
    cmd = [sys.executable, str(REPO / "scripts" / script), "--config", str(config), *extra]
    proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        raise SystemExit(f"SELF-TEST FAILED: {script} {' '.join(extra)} exited with {proc.returncode}")
    print(f"  ran {script} {' '.join(extra)}")


def check(condition: bool, what: str, failures: list[str]) -> None:
    print(f"  [{'ok' if condition else 'FAIL'}] {what}")
    if not condition:
        failures.append(what)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="keep the temp folder and print its path")
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="confdet_selftest_"))
    try:
        print(f"Building synthetic datasets in {work} ...")
        write_dataset(work / "raw" / "sim_dense", DEV_NAMES, 660, seed=1, video_share=0.5)
        inverse = [OTHER_TO_DEV.index(c) for c in range(len(DEV_NAMES))]       # dev class id -> OTHER class id
        write_dataset(work / "raw" / "sim_other", OTHER_NAMES, 300, seed=2, video_share=0.0, class_ids=inverse)
        config = make_config(work)

        print("Running the pipeline ...")
        run("02_prepare_data.py", work, config)
        run("03_compute_occlusion.py", work, config, "--strict")
        for det in ("sim_a", "sim_cautious", "sim_a_s1", "sim_a_s2"):
            run("04_train_detector.py", work, config, "--detector", det)
            run("05_predict.py", work, config, "--detector", det)
            run("06_calibrate.py", work, config, "--detector", det)
            run("07_run_evaluation.py", work, config, "--detector", det)
        run("15_build_ensemble.py", work, config, "--ensemble", "sim_a_ens=sim_a,sim_a_s1,sim_a_s2")
        run("06_calibrate.py", work, config, "--detector", "sim_a_ens")
        run("07_run_evaluation.py", work, config, "--detector", "sim_a_ens")
        run("08_compare_detectors.py", work, config, "--detectors", "sim_a", "sim_cautious", "sim_a_ens")
        run("14_seed_summary.py", work, config, "--family", "sim_a=sim_a,sim_a_s1,sim_a_s2")
        run("16_baselines.py", work, config, "--detectors", "sim_a", "sim_cautious", "sim_a_ens")
        run("13_calibrator_ablation.py", work, config, "--detectors", "sim_a", "sim_cautious")
        run("10_run_inference_images.py", work, config, "--detector", "sim_a", "--alpha", "0.1",
            "--images", str(work / "data" / "processed" / "selftest" / "sim_dense" / "images" / "val"),
            "--out-dir", str(work / "annotated"))
        run("12_summarize.py", work, config, "--detectors", "sim_a", "sim_cautious", "sim_a_ens")

        print("Checking the answers against what was built in ...")
        out = work / "outputs" / "selftest"
        load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
        fails: list[str] = []
        data = load(out / "data_report.json")
        res = out / "sim_a" / "class_agnostic" / "results"      # primary mode
        nat = out / "sim_a" / "native" / "results"
        s, rq1, rq2, rq3 = load(res / "summary.json"), load(res / "rq1.json"), load(res / "rq2.json"), load(res / "rq3.json")
        conf, rq5 = load(res / "conformal.json"), load(res / "rq5.json")
        rq4_agn, rq4_nat = load(res / "rq4_sim_other.json"), load(nat / "rq4_sim_other.json")
        det_agn, det_nat = load(res / "detection.json"), load(nat / "detection.json")
        rank = load(out / "comparison" / "class_agnostic" / "ranking.json")
        cal = load(out / "sim_a" / "class_agnostic" / "calibration.json")

        roles = data["roles"]["sim_dense"]
        check(roles["leaking_groups"] == [] and roles["n_video_segment_groups"] > 5,
              "splits: video segments and image copies never straddle roles", fails)
        check(all(roles["per_role"][r]["n_images"] > 0 for r in ("fit", "val", "calib", "test")),
              "splits: development dataset has fit / val / calib / test", fails)
        check(s["gates"]["detector_usable_on_development"]["usable"] and s["gates"]["enough_occlusion"]["adequate"],
              "gates: detector usable, enough occluded vehicles", fails)

        tau = cal["meta"]["standard_threshold"]
        check(0.05 <= tau <= 0.95 and s["standard_threshold"] == tau and cal["meta"]["matching"] == "class_agnostic",
              f"operating threshold chosen on calib by max F1 ({tau:.2f}) and carried through", fails)
        check(det_agn["this_mode"]["recall_at_floor"] > det_nat["this_mode"]["recall_at_floor"] + 0.02,
              "modes: misnamed vehicles are misses under native matching and found under class-agnostic "
              f"({det_nat['this_mode']['recall_at_floor']:.3f} vs {det_agn['this_mode']['recall_at_floor']:.3f})", fails)

        m = rq1["logistic_miss_model"]
        rates = [b["miss_rate"] for b in rq1["buckets"] if b["reliable"]]
        check(m["fitted"] and m["odds_ratio_ci"][0] > 1.0 and rates[-1] > 3 * rates[0],
              f"RQ1: misses rise with occlusion (odds x{m['odds_ratio_per_0.1_occlusion']:.2f} per +0.1)", fails)
        check(rq1["partial_spearman_ci"][0] > 0, "RQ1: effect survives controlling for size and class", fails)

        rep = conf["report"]
        target = rep["global_confidence"]["target_coverage"]
        check(rep["global_confidence"]["all_groups_achievable"] and
              rep["global_confidence"]["coverage"] >= target - 0.04,
              f"conformal: global rule delivers its target ({rep['global_confidence']['coverage']:.3f} vs {target})", fails)
        ok_groups = [g for g in rep["mondrian_confidence"]["by_context_group"] if g["achievable"]]
        check(bool(ok_groups) and all(g["coverage"] >= target - 0.07 for g in ok_groups),
              "conformal: occlusion-conditional rule delivers its target in every certifiable group", fails)
        g_top = rep["global_confidence"]["by_occlusion_bucket"][-1]["coverage"]
        m_top = rep["mondrian_confidence"]["by_occlusion_bucket"][-1]["coverage"]
        check(g_top < target - 0.1 and m_top > g_top,
              f"conformal: global rule under-covers the most occluded bucket ({g_top:.2f}); "
              f"conditional rule improves it ({m_top:.2f})", fails)

        ours = rq2["policies"]["occlusion_aware"]
        check(ours["prob_better_than_confidence"] >= 0.9 and ours["mean_recovery_delta_vs_confidence"] > 0,
              f"RQ2: duplicate- and context-aware score beats threshold-lowering at equal cost "
              f"(+{ours['mean_recovery_delta_vs_confidence']:.3f})", fails)
        check(all(row["recovery"] <= rq2["recoverable_ceiling"] + 1e-9 for row in ours["at_budgets"]),
              "RQ2: recovery never exceeds the recoverable ceiling", fails)

        p = rq3["at_standard_threshold"]
        check(rq3["identity_check"] and p["ma_ece"] > p["dece"] and
              abs(sum(b["share_of_gap"] for b in rq3["gap_decomposition_by_occlusion"]) - 1) < 1e-9,
              "RQ3: MA-ECE identity exact; gap decomposes exactly over occlusion buckets", fails)

        ct = rq4_nat["modes"]["coarse"]
        check(ct["gate"]["usable"] and ct["detection"]["recall_at_standard"] > 0.7,
              f"RQ4: classes matched by name through the common taxonomy "
              f"(recall {ct['detection']['recall_at_standard']:.2f} on the foreign-taxonomy dataset)", fails)
        check(rq4_agn["modes"]["class_agnostic"]["detection"]["recall_at_floor"] >=
              ct["detection"]["recall_at_floor"] - 1e-9, "RQ4: class-agnostic recall >= class-aware recall", fails)

        v = rq5["verdict"]
        check(bool(v) and v["visibility_misses_rise_with_occlusion"] and v["confusion_misses_rise_with_rarity"],
              "RQ5: visibility misses follow occlusion, confusion misses follow rarity", fails)

        own, matched = rank["at_own_threshold"], rank["at_matched_cost"]
        before = lambda order: order.index("sim_a") < order.index("sim_cautious")  # noqa: E731
        check(before(own["rankings"]["ma_ece"]) and before(own["rankings"]["oce_max"]),
              "ranking: the silent detector loses once misses count", fails)
        fps = [pt["fp_per_image"] for pt in matched["points"].values()]
        check(max(fps) - min(fps) <= 0.02 + 0.05 * matched["fp_per_image_budget"],
              f"ranking: matched-cost comparison really is at matched cost ({fps[0]:.2f} vs {fps[1]:.2f} "
              f"false alarms/image)", fails)
        digest = (out / "digest.md").read_text(encoding="utf-8")
        check("Matching: class_agnostic" in digest and "Matching: native" in digest and "sim_cautious" in digest,
              "digest written for both matching modes", fails)

        # v0.4 additions
        pair = matched["pairwise"][0]
        check(all("separates" in pair[m] for m in ("dece", "ma_ece", "ma_brier", "oce_max"))
              and all(not v or (pair["dece"]["separates"] and pair[m]["separates"])
                      for m, v in pair["flips_vs_dece"].items())
              and "flips vs D-ECE" in digest,
              "ranking: a flip is only counted where both intervals exclude zero; digest shows separation", fails)
        tr = rq4_agn["modes"]["class_agnostic"]["comparison"]
        by_a = tr.get("conformal_coverage_by_alpha", {})
        check("0.1" in by_a and all("transfer_recalibrated_flags_per_image" in r and
                                    "transfer_recalibrated_certifiable" in r
                                    for rows in by_a.values() for r in rows.values()),
              f"RQ4: rows at alpha {sorted(by_a)} with flags/image and certifiability", fails)
        check(len(rq1.get("by_size_tertile", [])) == 3 and "RQ1 by object size" in digest,
              "RQ1: miss rate by size tertile and occlusion x size in rq1.json and the digest", fails)
        abl_sel = load(out / "sim_a" / "class_agnostic" / "ablation_selection.json")
        abl_test = load(out / "sim_a" / "class_agnostic" / "ablation_test.json")
        check(abl_sel["chosen"] in abl_test["sets"] and abl_test["chosen_on_calib"] == abl_sel["chosen"]
              and abs(abl_test["sets"]["z_only"]["mean_recovery_delta_vs_confidence"]) < 1e-3
              and "RQ2 ablation" in digest,
              f"ablation: chosen on calib ({abl_sel['chosen']}), every set evaluated on test, z_only == confidence",
              fails)

        # v0.5 additions
        check(cal["meta"].get("score_features") == "full_strict" and cal["calibrator"]["name"] == "full_strict"
              and "duplicate- and context-aware score" in digest and "| occlusion_aware |" not in digest,
              "default score is full_strict and the digest calls it the duplicate- and context-aware score", fails)
        dec = rq3["ma_brier_decomposition_at_standard"]
        pts = [p["ma_brier_decomposition"] for blk in (own, matched) for p in blk["points"].values()]
        check(abs(dec["false_alarm"] + dec["matched"] + dec["miss"] - rq3["at_standard_threshold"]["ma_brier"]) < 1e-9
              and len(pts) == 2 * len(own["points"]) and "MA-Brier decomposition" in digest,
              "MA-Brier decomposed exactly (own threshold and matched cost) in rq3.json, ranking.json, digest", fails)
        seeds = load(out / "seeds" / "seed_summary.json")["families"]["sim_a"]
        ap = seeds["metrics"]["ap50_class_agnostic"]
        check(seeds["n_members"] == 3 and ap["n"] == 3 and ap["sd"] is not None and ap["sd"] > 0
              and seeds["metrics"]["conformal_a0.1.global_confidence.coverage"]["n"] == 3
              and "Training-seed variability" in digest,
              f"seeds: three seeds summarised (AP50 SD {ap['sd'] if ap['sd'] is not None else float('nan'):.4f}); "
              f"digest section written", fails)
        ens_det = load(out / "sim_a_ens" / "class_agnostic" / "results" / "detection.json")["this_mode"]
        ens_files = sorted(p.name for p in (out / "sim_a_ens" / "predictions").glob("*.npz"))
        check(ens_det["recall_at_floor"] > det_agn["this_mode"]["recall_at_floor"]
              and len(ens_files) == len(list((out / "sim_a" / "predictions").glob("*.npz")))
              and "sim_a_ens" in rank["at_own_threshold"]["detectors"],
              f"ensemble: built for {len(ens_files)} dataset/roles, finds more vehicles than one seed at the floor "
              f"({ens_det['recall_at_floor']:.3f} vs {det_agn['this_mode']['recall_at_floor']:.3f}), "
              f"runs through 06-08 unchanged", fails)

        # v0.6 additions: literature baselines on the same calib images as our rules
        bl = load(res / "baselines.json")
        rules = bl["by_alpha"]["0.1"]["rules"]
        crc = rules["crc_fnr"]
        check(crc["achievable"] and crc["mean_image_fnr"] <= 0.1 + 0.03
              and rules["pac"]["coverage"] >= 0.9 - 0.03
              and all(k in rules for k in ("seqcrc_box_count_recall", "global_gilg_iou_aware", "mondrian_confidence"))
              and bl["lrp"]["lrp_optimal_threshold_from_calib"]["lrp"] <= bl["lrp"]["operating_threshold_max_f1"]["lrp"] + 0.02
              and (out / "comparison" / "class_agnostic" / "baselines_summary.json").exists(),
              f"baselines: CRC per-image FNR {crc['mean_image_fnr']:.3f} <= 0.10, PAC coverage "
              f"{rules['pac']['coverage']:.3f}, SeqCRC/Gilg/LaECE0/LRP written for every detector", fails)
        check("Baselines from the literature" in digest and "LaECE_0" in digest and "IoU-aware calibration" in digest,
              "baselines: comparison table, Gilg vs ours and LaECE_0/LRP sections in the digest", fails)
        dets = load(work / "annotated" / "detections.json")
        check(len(dets) > 0 and sum(d["n_accepted"] for d in dets) > 0 and sum(d["n_flagged"] for d in dets) > 0
              and all(Path(d["annotated"]).exists() for d in dets),
              f"inference: {len(dets)} images annotated with accepted and flagged tiers", fails)
        figs = list((out / "sim_a" / "class_agnostic" / "figures").glob("*.png"))
        check(len(figs) >= 6 and (out / "comparison" / "class_agnostic" / "detectors.png").exists(),
              f"{len(figs)} figures written", fails)

        if fails:
            print(f"\nSELF-TEST FAILED ({len(fails)} check(s)):")
            for f in fails:
                print(f"  - {f}")
            return 1
        print("\nSELF-TEST PASSED: the pipeline runs end to end and recovers the known answers.")
        return 0
    finally:
        if args.keep:
            print(f"(kept {work})")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
