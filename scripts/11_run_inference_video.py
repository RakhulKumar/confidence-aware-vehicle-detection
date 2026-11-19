#!/usr/bin/env python3
"""Run detector + post-hoc layer over a video; optionally write an annotated
video and the N most useful annotated stills.

  python scripts/11_run_inference_video.py --video file.mp4 [--detector NAME]
         [--rule mondrian_confidence] [--frame-stride 5] [--write-video]
         [--best-frames 10] [--scorer flagged_showcase]
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from _common import setup

from confdet.data.splits import processed_info
from confdet.inference.extract_best_frames import SCORERS, save_best_frames
from confdet.inference.infer_video import run_inference_on_video, write_annotated_video
from confdet.models.detector_factory import detector_config, load_detector
from confdet.pipeline import RULES
from confdet.utils.paths import detector_dir, processed_root, weights_path

_spec = importlib.util.spec_from_file_location(
    "inference_images", Path(__file__).with_name("10_run_inference_images.py"))
_images = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_images)


def main():
    def extra(p):
        p.add_argument("--video", required=True)
        p.add_argument("--rule", default="mondrian_confidence", choices=list(RULES) + ["none"])
        p.add_argument("--alpha", type=float, default=None)
        p.add_argument("--frame-stride", type=int, default=5)
        p.add_argument("--max-frames", type=int, default=None)
        p.add_argument("--write-video", action="store_true")
        p.add_argument("--best-frames", type=int, default=10)
        p.add_argument("--scorer", default="flagged_showcase", choices=sorted(SCORERS))
        p.add_argument("--out-dir", default=None)

    args, cfg, log = setup("run_inference_video", __doc__, detector=True, extra=extra)
    det = detector_config(cfg, args.detector)
    names = processed_info(cfg["experiment"]["development"], processed_root(cfg)).names
    detector = load_detector(det, weights_path(cfg, det.name), cfg.get("prediction", {}), names,
                             gpu_share=cfg["detectors"].get("gpu_share"))
    out_dir = Path(args.out_dir) if args.out_dir else detector_dir(cfg, det.name) / "video" / Path(args.video).stem
    tau, kw = _images.tier_setup(cfg, det.name, args.rule, args.alpha)
    frames = run_inference_on_video(detector, args.video, tau, args.frame_stride, args.max_frames, **kw)
    n_acc = sum(sum(d.tier == "accepted" for d in f.detections) for f in frames)
    n_flag = sum(sum(d.tier == "flagged" for d in f.detections) for f in frames)
    log.info(f"{len(frames)} frames processed: {n_acc} accepted detections, {n_flag} flagged candidates")
    if args.write_video:
        write_annotated_video(args.video, frames, detector.names, out_dir / "annotated.mp4")
        log.info(f"annotated video: {out_dir / 'annotated.mp4'}")
    if args.best_frames > 0:
        saved = save_best_frames(args.video, frames, detector.names, out_dir, args.best_frames, args.scorer,
                                 min_gap_frames=max(15, 3 * args.frame_stride))
        log.info(f"{len(saved)} stills written to {out_dir}")


if __name__ == "__main__":
    main()
