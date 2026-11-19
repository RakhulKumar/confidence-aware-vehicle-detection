"""Draw the three-tier output on an image: accepted detections as solid red
boxes, flagged candidates ("possible occluded vehicle") as orange boxes with
their occlusion context, ground truth (if given) in green underneath so a
silently missed vehicle shows up as a bare green box.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_ACCEPTED, _FLAGGED, _GT = (220, 38, 38), (245, 158, 11), (34, 139, 34)


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def annotate_image(image, detections, class_names: list[str], ground_truth_boxes=None,
                   out_path: str | Path | None = None) -> Image.Image:
    """image: path or PIL image. detections: list of TieredDetection.
    ground_truth_boxes: optional iterable of (x1, y1, x2, y2).
    """
    img = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(max(12, img.width // 90))
    for box in ground_truth_boxes or []:
        draw.rectangle([float(v) for v in box[:4]], outline=_GT, width=2)
    for d in sorted(detections, key=lambda d: d.tier == "accepted"):      # accepted drawn last, on top
        name = class_names[d.class_id] if 0 <= d.class_id < len(class_names) else str(d.class_id)
        if d.tier == "accepted":
            color, width, label = _ACCEPTED, 3, f"{name} {d.confidence:.2f}"
        else:
            color, width, label = _FLAGGED, 2, f"? {name} {d.confidence:.2f} occ~{d.occlusion_context:.1f}"
        draw.rectangle([d.x1, d.y1, d.x2, d.y2], outline=color, width=width)
        tb = draw.textbbox((d.x1, d.y1), label, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color)
        draw.text((d.x1, d.y1), label, fill=(255, 255, 255), font=font)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
    return img
