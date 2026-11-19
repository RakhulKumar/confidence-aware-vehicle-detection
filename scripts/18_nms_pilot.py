#!/usr/bin/env python3
"""NMS pilot (v0.6 Phase 1b item 6), current YOLO26 (seed 0) only; does not touch the headline pipeline.

  python scripts/18_nms_pilot.py --predict [--cap 1000]     (GPU, under detectors.gpu_share)
      calib + test of the development dataset with suppression switched off (Ultralytics NMS at
      IoU 1.0 removes nothing), confidence >= the floor, the `cap` highest-confidence boxes per image.
      Writes outputs/<experiment>/<detector>/predictions_prenms/<dataset>__<role>.npz (skips files
      that already cover every image of their role).
  python scripts/18_nms_pilot.py --analyse                  (CPU)
      (i)  miss mechanism of every vehicle the standard output misses at its operating threshold,
           by annotated occlusion bucket (evaluation/nms.miss_mechanism);
      (ii) the cached standard output and, re-applied to the same pre-NMS boxes, hard NMS at IoU
           0.5 / 0.7 / 0.8 and Gaussian soft-NMS (sigma 0.5): AP50, and recall + coverage by
           annotated occlusion at matched false alarms per image (the standard output's own, and
           +0.5, +1.0 extra), class-agnostic.
      Writes outputs/<experiment>/<detector>/class_agnostic/results/nms_pilot.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from _common import setup

from confdet.calibration.detection_ece import operating_point
from confdet.data.occlusion import bucket_indices
from confdet.data.splits import processed_info, role_images
from confdet.evaluation.detection_metrics import detection_summary
from confdet.evaluation.nms import apply_suppression, miss_mechanism
from confdet.evaluation.records import load_records, save_records
from confdet.evaluation.rq3_calibration_gap import threshold_for_false_alarms
from confdet.models.detector_factory import UltralyticsDetector, apply_memory_fraction, detector_config
from confdet.pipeline import PostHoc, collect_records
from confdet.utils.io_utils import load_results, save_results
from confdet.utils.paths import calibration_path, detector_dir, predictions_path, processed_root, results_dir, weights_path

DET = "yolo26"


def _complete(path: Path, pairs) -> bool:
    if not path.exists():
        return False
    try:
        return sorted(r.image_id for r in load_records(path).records) == sorted(p.name for p, _ in pairs)
    except Exception:  # noqa: BLE001
        return False


def predict(cfg, cap: int, log) -> None:
    dev, root = cfg["experiment"]["development"], processed_root(cfg)
    det = detector_config(cfg, DET)
    share = cfg["detectors"].get("gpu_share") or {}
    apply_memory_fraction(share.get("memory_fraction"))
    detector = UltralyticsDetector(weights_path(cfg, DET), det.family, det.imgsz,
                                   float(cfg.get("prediction", {}).get("conf_floor", 0.001)),
                                   nms_iou=1.0, max_det=cap, duty_cycle=share.get("duty_cycle"))
    out_dir = detector_dir(cfg, DET) / "predictions_prenms"
    for role in ("calib", "test"):
        out = out_dir / f"{dev}__{role}.npz"
        pairs = role_images(dev, role, root)
        if _complete(out, pairs):
            log.info(f"{out} already complete; skipped")
            continue
        log.info(f"pre-NMS prediction {dev}/{role} with {DET} (IoU 1.0, cap {cap}) ...")
        rs = collect_records(detector, dev, role, cfg["occlusion"].get("proxy", "bottom_y"), root, log)
        rs.meta.update({"detector": DET, "nms_iou": 1.0, "max_det": cap, "note": "suppression switched off"})
        save_records(out, rs)
        n = sum(len(r.cand_conf) for r in rs.records)
        log.info(f"  {len(rs)} images, {n} candidates ({n / max(len(rs), 1):.0f} per image), "
                 f"{out.stat().st_size / 2**20:.1f} MB -> {out}")


def _variant_report(rs, tau, budgets, edges, cfg, from_tables=None) -> dict:
    from confdet.pipeline import tables_from_records
    t = from_tables or tables_from_records(cfg, rs, "class_agnostic", context_conf=tau)
    det = detection_summary(t, tau)
    rows = []
    for b in budgets:
        thr = threshold_for_false_alarms(t, b)
        p = operating_point(t, thr, with_oce=False)
        found = t.g_proposed & (t.g_score >= thr)
        bk = bucket_indices(t.g_occ, edges)
        rows.append({"fp_per_image_budget": b, "threshold": thr, "fp_per_image": p["fp_per_image"],
                     "recall": p["recall"],
                     "recall_by_annotated_occlusion": [float(found[bk == k].mean()) if (bk == k).any() else None
                                                       for k in range(len(edges) - 1)]})
    return {"ap50": det["ap50"], "recall_at_floor": det["recall_at_floor"],
            "candidates_per_image": t.n_cand / max(t.n_images, 1), "at_matched_false_alarms": rows}


def analyse(cfg, log) -> None:
    from confdet.pipeline import tables_from_records
    dev = cfg["experiment"]["development"]
    edges = cfg["occlusion"]["bucket_edges"]
    tau = PostHoc(load_results(calibration_path(cfg, DET, "class_agnostic"))).tau
    pre = load_records(detector_dir(cfg, DET) / "predictions_prenms" / f"{dev}__test.npz")
    std = load_records(predictions_path(cfg, DET, dev, "test"))
    if [r.image_id for r in pre.records] != [r.image_id for r in std.records]:
        raise SystemExit("pre-NMS and standard predictions cover different images")
    t_std = tables_from_records(cfg, std, "class_agnostic", context_conf=tau)
    mech = miss_mechanism(pre, t_std, tau, edges, float(cfg["evaluation"].get("iou_match_threshold", 0.5)))
    b0 = operating_point(t_std, tau, with_oce=False)["fp_per_image"]
    budgets = [b0, b0 + 0.5, b0 + 1.0]
    variants = {"standard_cached (Ultralytics NMS 0.7, max_det 300)": _variant_report(std, tau, budgets, edges, cfg, t_std)}
    for name, method, kw in (("hard_nms_0.5", "hard", {"iou_thr": 0.5}), ("hard_nms_0.7", "hard", {"iou_thr": 0.7}),
                             ("hard_nms_0.8", "hard", {"iou_thr": 0.8}), ("soft_nms_gaussian_0.5", "soft", {"sigma": 0.5})):
        variants[name] = _variant_report(apply_suppression(pre, method, **kw), tau, budgets, edges, cfg)
        r = variants[name]["at_matched_false_alarms"][0]
        log.info(f"  {name:<24} AP50 {variants[name]['ap50']:.4f} | at {b0:.2f} FA/img: recall {r['recall']:.4f} "
                 f"by o_j {['%.3f' % x for x in r['recall_by_annotated_occlusion']]}")
    out = {"detector": DET, "tau": tau, "standard_fp_per_image": b0,
           "prenms_candidates_per_image": sum(len(r.cand_conf) for r in pre.records) / len(pre.records),
           "prenms_cap_per_image": pre.meta.get("max_det"), "miss_mechanism_test": mech, "variants": variants,
           "suppression_is_per_class": True}
    save_results(out, results_dir(cfg, DET, "class_agnostic") / "nms_pilot.json")
    log.info(f"miss mechanism (test, tau {tau:.2f}, {mech['n_missed']} misses): "
             + ", ".join(f"{k} {v:.3f}" for k, v in mech["shares"].items()))


def main():
    def extra(p):
        p.add_argument("--predict", action="store_true")
        p.add_argument("--analyse", action="store_true")
        p.add_argument("--cap", type=int, default=1000)

    args, cfg, log = setup("nms_pilot", __doc__, extra=extra)
    if args.predict:
        predict(cfg, args.cap, log)
    if args.analyse:
        analyse(cfg, log)


if __name__ == "__main__":
    main()
