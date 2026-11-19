#!/usr/bin/env python3
"""Stage 0: validate the raw datasets, then build leak-free pipeline roles.

  python scripts/02_prepare_data.py --check-only    # validate only
  python scripts/02_prepare_data.py                 # validate + split + materialise

Writes data/processed/<experiment>/<dataset>/ (plain-bbox labels, hard-linked
images, roles fit / val / calib / test) and outputs/<experiment>/data_report.json.
Re-running replaces the materialised copy. Run this FIRST, and again whenever
a dataset or its `roles` rules change.
"""
from __future__ import annotations

import sys

from _common import setup

from confdet.data.splits import build_dataset_roles
from confdet.data.yolo_dataset import load_dataset_info, validate_dataset
from confdet.utils.io_utils import experiment_datasets, save_results
from confdet.utils.paths import experiment_dir, processed_root


def main():
    args, cfg, log = setup("prepare_data", __doc__,
                           extra=lambda p: p.add_argument("--check-only", action="store_true"))
    if cfg.get("derive") or cfg.get("materialised_by"):
        raise SystemExit(f"{args.config} has a `derive:` block: its data is built from an existing split by "
                         f"scripts/02b_derive_experiment.py --build. scripts/02 would re-split it and is refused.")
    dev = cfg["experiment"]["development"]
    report, ok = {"validation": {}, "roles": {}}, True

    for name in experiment_datasets(cfg):
        ds = cfg["datasets"][name]
        log.info(f"Validating {name} at {ds['root']} (label_format={ds.get('label_format', 'bbox')}) ...")
        try:
            info = load_dataset_info(ds["root"], ds.get("label_format", "bbox"))
        except FileNotFoundError as e:
            log.error(f"  {e}")
            report["validation"][name] = {"ok": False, "error": str(e)}
            ok = False
            continue
        wanted = sorted({s for rule in ds.get("roles", []) for s in rule["from"]})
        per_split = {}
        for split in wanted:
            if split not in info.split_dirs:
                log.error(f"  roles rule asks for split {split!r}; data.yaml defines {list(info.split_dirs)}")
                ok = False
                continue
            rep = validate_dataset(info, split)
            per_split[split] = rep
            shapes = rep["label_line_shapes"]
            log.info(f"  [{'OK' if rep['ok'] else 'ISSUES'}] {split}: {rep['n_images']} images, "
                     f"{rep['n_boxes']} boxes ({rep['boxes_per_image']:.2f}/image), "
                     f"{rep['n_images_without_boxes']} without boxes; lines: {shapes['bbox_lines']} box, "
                     f"{shapes['polygon_lines']} polygon")
            for issue in rep["issues"]:
                log.error(f"      {issue}")
            ok = ok and rep["ok"]
        report["validation"][name] = {"ok": all(r["ok"] for r in per_split.values()), "names": info.names,
                                      "splits": per_split}

    out_path = experiment_dir(cfg) / "data_report.json"
    if not ok:
        save_results(report, out_path)
        log.error("Dataset problems found (see above). Fix them at the source or in the config, "
                  "then re-run. Nothing was materialised.")
        sys.exit(1)
    if args.check_only:
        save_results(report, out_path)
        log.info("All datasets valid. Re-run without --check-only to build the roles.")
        return

    seed = int(cfg["project"].get("seed", 42))
    for name in experiment_datasets(cfg):
        log.info(f"Splitting and materialising {name} ...")
        summary = build_dataset_roles(name, cfg["datasets"][name], seed, cfg["occlusion"], processed_root(cfg))
        report["roles"][name] = summary
        for role, s in summary["per_role"].items():
            log.info(f"  {role:<5} {s['n_images']:>6} images  {s['n_objects']:>7} objects  "
                     f"{s['n_occluded']:>6} occluded  ({s['n_groups']} groups)")
        log.info(f"  {summary['n_groups']} leakage groups, largest {summary['largest_groups']}, "
                 f"{summary['n_video_segment_groups']} video segments; e.g. {summary['example_files'][:2]}")
        if summary["leaking_groups"]:
            log.error(f"  groups present in more than one role: {summary['leaking_groups'][:5]}")
            ok = False
        need = ("fit", "val", "calib", "test") if name == dev else ("test",)
        missing = [r for r in need if r not in summary["per_role"] or summary["per_role"][r]["n_images"] == 0]
        if missing:
            log.error(f"  {name} is the {'development' if name == dev else 'transfer'} dataset and needs "
                      f"role(s) {missing}; adjust its `roles` rules")
            ok = False
    save_results(report, out_path)
    log.info(f"Report: {out_path}")
    if not ok:
        sys.exit(1)
    log.info("Done. Next: python scripts/03_compute_occlusion.py")


if __name__ == "__main__":
    main()
