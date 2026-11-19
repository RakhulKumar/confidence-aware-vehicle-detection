#!/usr/bin/env python3
"""Stage 1: fine-tune one detector on the development dataset's fit role
(model selection on its val role). The calib and test roles are never shown
to the detector.

  python scripts/04_train_detector.py                    # detectors.active
  python scripts/04_train_detector.py --detector yolov8
  python scripts/04_train_detector.py --detector rtdetr
  python scripts/04_train_detector.py --detector yolo26 --batch 16   # override the config's batch (Beocat L40S)

The best checkpoint always ends up at
outputs/<experiment>/<detector>/weights/best.pt, whatever Ultralytics names
its run folder. Training stops with an explanation if the batch overflows
GPU memory.
"""
from __future__ import annotations

import dataclasses

from _common import setup

from confdet.data.splits import has_role
from confdet.models.detector_factory import detector_config, last_checkpoint, train
from confdet.utils.paths import processed_root, train_dir, weights_path


def main():
    def extra(p):
        p.add_argument("--resume", action="store_true",
                       help="continue from <run>/train/weights/last.pt if it exists (restart-safe)")
        p.add_argument("--retrain", action="store_true",
                       help="train again although this run already finished (its best.pt exists)")
        p.add_argument("--batch", type=int, default=None,
                       help="batch size instead of the config's (one value per architecture on new hardware; "
                            "a resumed run keeps the batch stored in its last.pt)")

    args, cfg, log = setup("train_detector", __doc__, detector=True, extra=extra)
    dev = cfg["experiment"]["development"]
    root = processed_root(cfg)
    for role in ("fit", "val"):
        if not has_role(dev, role, root):
            raise SystemExit(f"{root / dev} has no {role!r} role -- run scripts/02_prepare_data.py first")
    det = detector_config(cfg, args.detector)
    if args.batch is not None:
        if args.batch < 1:
            raise SystemExit(f"--batch must be positive, got {args.batch}")
        log.info(f"batch {det.batch} (config) -> {args.batch} (--batch)")
        det = dataclasses.replace(det, batch=args.batch)
    data_yaml = root / dev / "data.yaml"
    run_dir = train_dir(cfg, det.name)
    # A finished run whose rolling last.pt was cleaned up must not silently start again from scratch.
    if args.resume and not args.retrain and weights_path(cfg, det.name).exists() and not last_checkpoint(run_dir).exists():
        log.info(f"{weights_path(cfg, det.name)} exists and the run has no last.pt (finished and cleaned up): "
                 f"nothing to do. Pass --retrain to train again.")
        return
    log.info(f"Training {det.name} ({det.family}, {det.weights}) on {data_yaml}: "
             f"imgsz={det.imgsz} batch={det.batch} epochs={det.epochs} patience={det.patience}")
    best = train(det, data_yaml, run_dir, device=cfg["project"].get("device", "cuda"),
                 seed=int(cfg["project"].get("seed", 42)),
                 abort_on_vram_overflow=bool(cfg["detectors"].get("abort_on_vram_overflow", True)),
                 logger=log, resume=args.resume, gpu_share=cfg["detectors"].get("gpu_share"))
    log.info(f"Checkpoint: {best}")
    log.info(f"Next: python scripts/05_predict.py --detector {det.name}")


if __name__ == "__main__":
    main()
