#!/usr/bin/env python3
"""Cross-check: run Ultralytics' OWN validator on the development TEST role.

  python scripts/check_ultralytics_val.py [--detector NAME]

Why: in the first real run Ultralytics reported mAP50 0.75 on the val role
during training, while this project's class-aware AP50 on the test role was
0.67 for the YOLO models (0.73 for RT-DETR). Two explanations are possible
and this tells them apart in a few minutes:

  * Ultralytics also reports ~0.67 on test  -> the val number was optimistic
    (val was used to pick the best epoch) or the two splits differ;
  * Ultralytics reports ~0.75 on test       -> the gap comes from how
    candidates are produced. Ultralytics' validator keeps several class
    hypotheses per box; predict() keeps the best class only, so a vehicle
    whose second-best class is the right one has no correctly-classed
    candidate here. That affects only class-aware (native) numbers and is
    one more reason the class-agnostic mode is the primary one.

Writes outputs/<experiment>/<detector>/ultralytics_val_test.json.
"""
from __future__ import annotations

from _common import setup

from confdet.models.detector_factory import _ultralytics_class, detector_config
from confdet.utils.io_utils import save_results
from confdet.utils.paths import detector_dir, processed_root, weights_path


def main():
    args, cfg, log = setup("check_ultralytics_val", __doc__, detector=True)
    det = detector_config(cfg, args.detector)
    if det.family == "simulated":
        raise SystemExit("needs a real Ultralytics detector")
    dev = cfg["experiment"]["development"]
    data_yaml = (processed_root(cfg) / dev / "data.yaml").resolve()
    weights = weights_path(cfg, det.name)
    if not weights.exists():
        raise SystemExit(f"{weights} not found")
    model = _ultralytics_class(det.family)(str(weights))
    out_dir = detector_dir(cfg, det.name).resolve()
    metrics = model.val(data=str(data_yaml), split="test", imgsz=det.imgsz, batch=det.batch,
                        conf=float(cfg.get("prediction", {}).get("conf_floor", 0.001)),
                        iou=float(cfg.get("prediction", {}).get("nms_iou", 0.7)),
                        max_det=int(cfg.get("prediction", {}).get("max_det", 300)),
                        project=str(out_dir), name="ultralytics_val_test", exist_ok=True, plots=False)
    box = getattr(metrics, "box", None)
    result = {"detector": det.name, "split": "test",
              "map50": float(getattr(box, "map50", float("nan"))) if box is not None else None,
              "map50_95": float(getattr(box, "map", float("nan"))) if box is not None else None,
              "mean_precision": float(getattr(box, "mp", float("nan"))) if box is not None else None,
              "mean_recall": float(getattr(box, "mr", float("nan"))) if box is not None else None}
    save_results(result, out_dir / "ultralytics_val_test.json")
    log.info(f"Ultralytics validator on {dev}/test: mAP50 {result['map50']}, mAP50-95 {result['map50_95']}")
    log.info("Compare mAP50 with `class_aware.ap50` in <detector>/native/results/detection.json.")


if __name__ == "__main__":
    main()
