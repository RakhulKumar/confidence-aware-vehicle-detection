#!/usr/bin/env python3
"""Stage 2: occlusion severity of every annotated vehicle, per dataset and
role, and the POWER GATE: can this data answer occlusion-stratified
questions at all?

  python scripts/03_compute_occlusion.py [--strict]

Writes outputs/<experiment>/occlusion_report.json and
outputs/<experiment>/figures/occlusion_histogram.png. No GPU, takes seconds:
look at this table before training anything. With --strict the script exits
non-zero if the development test role is underpowered.
"""
from __future__ import annotations

import sys

import numpy as np
from _common import setup

from confdet.data.occlusion import bucket_indices, occlusion_severity_array
from confdet.data.splits import ROLES, has_role, role_images
from confdet.data.yolo_dataset import image_size, read_label_file
from confdet.evaluation.rq1_correlation import power_check
from confdet.utils.io_utils import experiment_datasets, save_results
from confdet.utils.paths import experiment_dir, processed_root
from confdet.visualization.plots import plot_occlusion_histogram, save


def main():
    args, cfg, log = setup("compute_occlusion", __doc__,
                           extra=lambda p: p.add_argument("--strict", action="store_true"))
    occ_cfg, ev = cfg["occlusion"], cfg["evaluation"]
    edges, thr = occ_cfg["bucket_edges"], float(occ_cfg.get("occluded_threshold", 0.2))
    need = int(ev.get("min_occluded_objects", 200))
    root = processed_root(cfg)
    dev = cfg["experiment"]["development"]
    report, dev_ok = {}, True
    # The gate is about the development dataset's calib and test roles. If they are not on disk
    # (data moved or deleted), stop BEFORE writing: an empty report must never pass the gate or
    # overwrite a real one.
    missing = [r for r in ("calib", "test") if not has_role(dev, r, root)]
    if missing:
        sys.exit(f"the development dataset {dev!r} has no {missing} role under {root / dev}. "
                 f"Is project.processed_dir right, and is the materialised data still there? "
                 f"Do NOT re-run scripts/02 to fix this unless you intend to re-split (that invalidates "
                 f"every checkpoint and cached prediction). Nothing was written.")

    for name in experiment_datasets(cfg):
        for role in ROLES:
            if not has_role(name, role, root):
                continue
            sev, n_images, n_objects_per_image = [], 0, []
            for image_path, label_path in role_images(name, role, root):
                n_images += 1
                rows, _ = read_label_file(label_path, "bbox")
                n_objects_per_image.append(len(rows))
                if len(rows) >= 2:
                    w, h = image_size(image_path)
                    xyxy = np.array([[r[1] * w, r[2] * h, r[3] * w, r[4] * h] for r in rows])
                    sev.extend(occlusion_severity_array(xyxy, occ_cfg.get("proxy", "bottom_y")).tolist())
                else:
                    sev.extend([0.0] * len(rows))
            sev = np.array(sev)
            b = bucket_indices(sev, edges) if len(sev) else np.zeros(0, dtype=int)
            power = power_check(sev, thr, need)
            entry = {
                "n_images": n_images, "n_objects": int(len(sev)),
                "objects_per_image": float(np.mean(n_objects_per_image)) if n_images else 0.0,
                "share_images_without_objects": float(np.mean(np.array(n_objects_per_image) == 0)) if n_images else 0.0,
                "buckets": [{"lo": edges[k], "hi": edges[k + 1], "n": int((b == k).sum())}
                            for k in range(len(edges) - 1)],
                "n_occluded": power["n_occluded"], "adequate": power["adequate"], "message": power["message"],
            }
            report[f"{name}/{role}"] = entry
            counts = "  ".join(f"{x['n']:>6}" for x in entry["buckets"])
            log.info(f"{name + '/' + role:<22} {n_images:>6} img {entry['objects_per_image']:>5.1f} obj/img | "
                     f"per bucket: {counts} | occluded: {power['n_occluded']:>6} "
                     f"{'OK' if power['adequate'] else 'UNDERPOWERED'}")
            if name == dev and role in ("calib", "test") and not power["adequate"]:
                dev_ok = False

    verdict = {
        "development_dataset": dev,
        "development_can_answer_occlusion_questions": dev_ok,
        "rule": f">= {need} objects with occlusion severity >= {thr} in each of calib and test",
    }
    out = experiment_dir(cfg)
    save_results({"per_role": report, "verdict": verdict}, out / "occlusion_report.json")
    if report:
        save(plot_occlusion_histogram(report), out / "figures" / "occlusion_histogram.png")
    log.info(f"Report: {out / 'occlusion_report.json'}")
    if dev_ok:
        log.info(f"POWER GATE: {dev} has enough occluded vehicles for RQ1-RQ3. "
                 f"Next: python scripts/04_train_detector.py")
    else:
        log.warning(
            f"POWER GATE FAILED: the development dataset {dev!r} does not contain enough occluded "
            f"vehicles ({verdict['rule']}). Occlusion-stratified results (RQ1, RQ2 on occluded misses, "
            f"RQ3 by bucket, RQ5) from this experiment will be flagged as not interpretable. Use a "
            f"dataset with real vehicle-on-vehicle occlusion as `development` (configs/experiment.yaml).")
        if args.strict:
            sys.exit(2)


if __name__ == "__main__":
    main()
