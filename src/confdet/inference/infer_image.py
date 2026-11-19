"""Single-image inference with the three-tier decision rule."""
from __future__ import annotations

from pathlib import Path

from confdet.inference.tiers import TieredDetection, assign_tiers
from confdet.visualization.annotate import annotate_image


def infer_image(detector, source, std_threshold: float, thresholds=None, calibrator=None,
                context_conf: float | None = None, duplicate_iou: float = 0.5,
                proxy: str = "bottom_y") -> list[TieredDetection]:
    """source: image path or HxWx3 BGR array."""
    xyxy, conf, cls = detector.predict(source)
    return assign_tiers(xyxy, conf, cls, std_threshold, thresholds, calibrator,
                        context_conf, duplicate_iou, proxy)


def infer_image_batch(detector, image_paths, output_dir: str | Path, std_threshold: float,
                      **kwargs) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for image_path in image_paths:
        image_path = Path(image_path)
        dets = infer_image(detector, image_path, std_threshold, **kwargs)
        out_path = output_dir / f"{image_path.stem}_annotated.jpg"
        annotate_image(image_path, dets, detector.names, out_path=out_path)
        results.append({"image": str(image_path), "annotated": str(out_path),
                        "n_accepted": sum(d.tier == "accepted" for d in dets),
                        "n_flagged": sum(d.tier == "flagged" for d in dets),
                        "detections": [d.__dict__ for d in dets]})
    return results
