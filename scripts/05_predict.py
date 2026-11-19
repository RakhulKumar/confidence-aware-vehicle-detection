#!/usr/bin/env python3
"""Run the frozen detector once over every image the analysis needs and
cache the raw candidates (confidence >= prediction.conf_floor):

  development dataset: calib, test
  transfer datasets:   calib (if present), test

  python scripts/05_predict.py [--detector NAME] [--limit N]

Everything after this stage reads these caches and needs no GPU.
"""
from __future__ import annotations

from _common import setup

from confdet.data.splits import has_role, processed_info, role_images
from confdet.evaluation.records import load_records, save_records
from confdet.models.detector_factory import detector_config, load_detector
from confdet.pipeline import collect_records
from confdet.utils.io_utils import experiment_datasets
from confdet.utils.paths import predictions_path, processed_root, weights_path


def prediction_file_complete(path, pairs) -> bool:
    """A cached prediction file counts as done only if it loads and covers
    exactly the images of its role (restart safety: a partial or stale file is redone)."""
    if not path.exists():
        return False
    try:
        rs = load_records(path)
    except Exception:
        return False
    return sorted(r.image_id for r in rs.records) == sorted(p.name for p, _ in pairs)


def main():
    def extra(p):
        p.add_argument("--limit", type=int, default=None, help="only the first N images per role (smoke test)")
        p.add_argument("--force", action="store_true", help="re-predict files that already exist and verify")

    args, cfg, log = setup("predict", __doc__, detector=True, extra=extra)
    root = processed_root(cfg)
    dev = cfg["experiment"]["development"]
    det = detector_config(cfg, args.detector)
    detector = load_detector(det, weights_path(cfg, det.name), cfg.get("prediction", {}),
                             processed_info(dev, root).names, gpu_share=cfg["detectors"].get("gpu_share"))
    for name in experiment_datasets(cfg):
        for role in ("calib", "test"):
            if not has_role(name, role, root):
                continue
            out = predictions_path(cfg, det.name, name, role)
            if not args.force and args.limit is None and prediction_file_complete(out, role_images(name, role, root)):
                log.info(f"{out} already complete; skipped (use --force to redo)")
                continue
            log.info(f"Predicting {name}/{role} with {det.name} ...")
            rs = collect_records(detector, name, role, cfg["occlusion"].get("proxy", "bottom_y"),
                                 root, log, args.limit)
            rs.meta.update({"detector": det.name, "conf_floor": cfg.get("prediction", {}).get("conf_floor", 0.001)})
            save_records(out, rs)
            n_c = sum(len(r.cand_conf) for r in rs.records)
            n_g = sum(len(r.gt_cls) for r in rs.records)
            log.info(f"  {len(rs)} images, {n_g} annotated vehicles, {n_c} raw candidates -> {out}")
    log.info(f"Next: python scripts/06_calibrate.py --detector {det.name}")


if __name__ == "__main__":
    main()
