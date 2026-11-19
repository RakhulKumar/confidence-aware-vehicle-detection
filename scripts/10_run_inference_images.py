#!/usr/bin/env python3
"""Run a trained detector plus the fitted post-hoc layer on your own images
and save annotated copies: accepted detections in red, flagged candidates
("possible occluded vehicle") in orange.

  python scripts/10_run_inference_images.py --images path/to/folder [--detector NAME]
         [--rule mondrian_confidence] [--alpha 0.1] [--out-dir DIR]

--rule none gives the plain thresholded detector output.
"""
from __future__ import annotations

from pathlib import Path

from _common import setup

from confdet.data.splits import processed_info
from confdet.inference.infer_image import infer_image_batch
from confdet.models.detector_factory import detector_config, load_detector
from confdet.pipeline import RULES, PostHoc
from confdet.utils.io_utils import load_results, matching_modes, save_results
from confdet.utils.paths import calibration_path, detector_dir, processed_root, weights_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def tier_setup(cfg, detector_name: str, rule: str, alpha: float | None, mode: str | None = None):
    """(operating threshold, keyword arguments for inference.tiers.assign_tiers) from the
    post-hoc layer fitted for `mode` (default: the primary matching mode).
    """
    mode = mode or matching_modes(cfg)[0]
    posthoc = PostHoc(load_results(calibration_path(cfg, detector_name, mode)))
    kw = {"context_conf": posthoc.context_conf,
          "duplicate_iou": float(cfg["posthoc"].get("duplicate_iou", 0.5)),
          "proxy": cfg["occlusion"].get("proxy", "bottom_y")}
    if rule != "none":
        rules = posthoc.rules(alpha)
        if rule not in rules:
            raise SystemExit(f"rule {rule!r} was not fitted (available: {sorted(rules)})")
        kw.update({"thresholds": rules[rule], "calibrator": posthoc.calibrator})
    return posthoc.tau, kw


def main():
    def extra(p):
        p.add_argument("--images", required=True, help="folder of images, or one image")
        p.add_argument("--rule", default="mondrian_confidence", choices=list(RULES) + ["none"])
        p.add_argument("--alpha", type=float, default=None, help="default: posthoc.alpha")
        p.add_argument("--out-dir", default=None)

    args, cfg, log = setup("run_inference_images", __doc__, detector=True, extra=extra)
    det = detector_config(cfg, args.detector)
    names = processed_info(cfg["experiment"]["development"], processed_root(cfg)).names
    detector = load_detector(det, weights_path(cfg, det.name), cfg.get("prediction", {}), names,
                             gpu_share=cfg["detectors"].get("gpu_share"))
    src = Path(args.images)
    paths = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"no images found at {src}")
    out_dir = Path(args.out_dir) if args.out_dir else detector_dir(cfg, det.name) / "annotated_images"
    tau, kw = tier_setup(cfg, det.name, args.rule, args.alpha)
    results = infer_image_batch(detector, paths, out_dir, tau, **kw)
    for r in results:
        log.info(f"  {Path(r['image']).name}: {r['n_accepted']} accepted, {r['n_flagged']} flagged")
    save_results(results, out_dir / "detections.json")
    log.info(f"Annotated images and detections.json in {out_dir}")


if __name__ == "__main__":
    main()
