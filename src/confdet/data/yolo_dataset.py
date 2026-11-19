"""YOLO-txt dataset loading.

Two on-disk layouts are read transparently:
  * Roboflow export   <split>/images/*.jpg + <split>/labels/*.txt
                      (data.yaml values like "../train/images"; the folder for
                      the "val" key is usually named "valid")
  * type-first        images/<split>/*.jpg + labels/<split>/*.txt

Label CONTENT format is never guessed per dataset. It is an explicit
per-dataset setting, `label_format`, in configs/experiment.yaml:

  * "bbox"     every line is exactly "<class> <cx> <cy> <w> <h>"
  * "polygon"  each line is EITHER that 5-field box OR a Roboflow polygon
               "<class> <x1> <y1> ... <xn> <yn>" (n >= 3); a polygon is
               converted to its axis-aligned envelope.

A "bbox" dataset that contains a polygon-shaped line is an error, reported
with the file name, rather than something to be silently reinterpreted.

All coordinates are clipped to the image; boxes that are empty after clipping
are dropped and counted. Dependency-light on purpose (yaml + Pillow).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from confdet.data.occlusion import Box
from confdet.utils.paths import long_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_FORMATS = ("bbox", "polygon")


@dataclass(frozen=True)
class DatasetInfo:
    root: Path
    names: list[str]
    nc: int
    split_dirs: dict[str, str]      # data.yaml key -> raw path string
    label_format: str = "bbox"


class LabelFormatError(ValueError):
    """A label line does not have the shape the dataset's label_format allows."""


def load_dataset_info(root: str | Path, label_format: str = "bbox") -> DatasetInfo:
    if label_format not in LABEL_FORMATS:
        raise ValueError(f"label_format must be one of {LABEL_FORMATS}, got {label_format!r}")
    root = Path(root)
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        present = sorted(p.name for p in root.parent.iterdir()) if root.parent.exists() else []
        raise FileNotFoundError(
            f"{data_yaml} not found. Folders present in {root.parent}: {present}. "
            f"Point the dataset's `root` in configs/experiment.yaml at the folder "
            f"that contains data.yaml."
        )
    with open(data_yaml, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    names = spec["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    names = [str(n) for n in names]
    split_dirs = {k: str(spec[k]) for k in ("train", "val", "test") if spec.get(k)}
    return DatasetInfo(root=root, names=names, nc=int(spec.get("nc", len(names))),
                       split_dirs=split_dirs, label_format=label_format)


# ---------------------------------------------------------------------------
# Label parsing (normalised coordinates in, normalised xyxy out)
# ---------------------------------------------------------------------------
def parse_label_line(line: str, label_format: str) -> tuple[int, tuple[float, float, float, float], str]:
    """Returns (class_id, (x1, y1, x2, y2) normalised and unclipped, shape)
    where shape is "bbox" or "polygon". Raises LabelFormatError otherwise.
    """
    parts = line.split()
    try:
        class_id = int(float(parts[0]))
        if float(parts[0]) != class_id:
            raise ValueError
        coords = [float(p) for p in parts[1:]]
    except (ValueError, IndexError):
        raise LabelFormatError(f"unparseable label line: {line!r}") from None
    if len(coords) == 4:
        cx, cy, w, h = coords
        return class_id, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), "bbox"
    if len(coords) >= 6 and len(coords) % 2 == 0:
        if label_format != "polygon":
            raise LabelFormatError(
                f"polygon-shaped line ({len(coords)} coordinates) in a dataset declared "
                f"label_format: bbox. If this dataset was annotated with polygons, set "
                f"label_format: polygon for it in configs/experiment.yaml. Line: {line[:80]!r}"
            )
        xs, ys = coords[0::2], coords[1::2]
        return class_id, (min(xs), min(ys), max(xs), max(ys)), "polygon"
    raise LabelFormatError(
        f"expected 4 numbers (box) or an even count >= 6 (polygon) after the class id, "
        f"got {len(coords)}: {line[:80]!r}"
    )


def _clip01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def read_label_file(label_path: Path, label_format: str, strict: bool = True):
    """Parse one label file.

    Returns (rows, stats): rows is a list of (class_id, x1, y1, x2, y2) in
    normalised, clipped coordinates; stats counts line shapes and problems.
    With strict=False malformed lines are counted instead of raised.
    """
    stats = {"bbox_lines": 0, "polygon_lines": 0, "bad_lines": 0, "empty_after_clip": 0}
    rows: list[tuple[int, float, float, float, float]] = []
    label_path = long_path(label_path)       # a 260+ character path would read as "no label file"
    if not label_path.exists():
        return rows, stats
    with open(label_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                class_id, (x1, y1, x2, y2), shape = parse_label_line(line, label_format)
            except LabelFormatError as e:
                if strict:
                    raise LabelFormatError(f"{label_path}: {e}") from None
                stats["bad_lines"] += 1
                continue
            stats[f"{shape}_lines"] += 1
            x1, y1, x2, y2 = _clip01(x1), _clip01(y1), _clip01(x2), _clip01(y2)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                stats["empty_after_clip"] += 1
                continue
            rows.append((class_id, x1, y1, x2, y2))
    return rows, stats


def rows_to_boxes(rows, img_w: int, img_h: int) -> list[Box]:
    return [Box(x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h, class_id=c)
            for c, x1, y1, x2, y2 in rows]


def rows_to_yolo_lines(rows) -> list[str]:
    """Plain "<class> <cx> <cy> <w> <h>" lines, 6 decimals."""
    out = []
    for c, x1, y1, x2, y2 in rows:
        out.append(f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return out


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------
def _split_path_parts(raw_path: str) -> list[str]:
    """data.yaml path components without '..' / '.' (Roboflow writes
    '../train/images' relative to a directory that is not the dataset root).
    """
    return [p for p in Path(raw_path).parts if p not in ("..", ".")]


def split_directories(info: DatasetInfo, split: str) -> tuple[Path, Path]:
    """(images_dir, labels_dir) for one data.yaml split key."""
    if split not in info.split_dirs:
        raise ValueError(f"Split {split!r} not in {info.root / 'data.yaml'} (have: {list(info.split_dirs)})")
    parts = _split_path_parts(info.split_dirs[split])
    if "images" not in parts:
        raise ValueError(
            f"data.yaml entry for {split!r} ({info.split_dirs[split]!r}) has no 'images' "
            f"path component, so the labels directory cannot be derived. Supported: "
            f"'<split>/images' and 'images/<split>'."
        )
    images_dir = (info.root / Path(*parts)).resolve()
    label_parts = list(parts)
    label_parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    labels_dir = (info.root / Path(*label_parts)).resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    return images_dir, labels_dir


def list_split_images(info: DatasetInfo, split: str) -> list[tuple[Path, Path]]:
    """Sorted [(image_path, label_path), ...] for a split."""
    images_dir, labels_dir = split_directories(info, split)
    pairs = []
    for image_path in sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        rel = image_path.relative_to(images_dir)
        pairs.append((image_path, labels_dir / rel.with_suffix(".txt")))
    return pairs


def image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image  # lazy: keeps this module importable with yaml only

    with Image.open(long_path(image_path)) as im:
        return im.size


def iter_split(info: DatasetInfo, split: str):
    """Yield (image_path, [Box, ...]) in pixel coordinates for every image."""
    for image_path, label_path in list_split_images(info, split):
        w, h = image_size(image_path)
        rows, _ = read_label_file(label_path, info.label_format, strict=True)
        yield image_path, rows_to_boxes(rows, w, h)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dataset(info: DatasetInfo, split: str) -> dict:
    """Structural checks for one split. Never raises on bad label content;
    reports it.
    """
    n_images = n_boxes = n_empty = n_missing_label = 0
    out_of_range = 0
    totals = {"bbox_lines": 0, "polygon_lines": 0, "bad_lines": 0, "empty_after_clip": 0}
    class_counts = [0] * info.nc
    first_bad: str | None = None

    for image_path, label_path in list_split_images(info, split):
        n_images += 1
        if not long_path(label_path).exists():
            n_missing_label += 1
        rows, stats = read_label_file(label_path, info.label_format, strict=False)
        if stats["bad_lines"] and first_bad is None:
            first_bad = str(label_path)
        for k in totals:
            totals[k] += stats[k]
        if not rows:
            n_empty += 1
        for c, *_ in rows:
            n_boxes += 1
            if 0 <= c < info.nc:
                class_counts[c] += 1
            else:
                out_of_range += 1

    issues: list[str] = []
    if n_images == 0:
        issues.append("no images found for this split")
    if out_of_range:
        issues.append(f"{out_of_range} boxes with class id outside [0, {info.nc})")
    if totals["bad_lines"]:
        hint = (" -- these look like polygon lines; set label_format: polygon for this dataset"
                if info.label_format == "bbox" else "")
        issues.append(f"{totals['bad_lines']} label lines do not match label_format="
                      f"{info.label_format!r}{hint} (first file: {first_bad})")
    return {
        "split": split, "n_images": n_images, "n_boxes": n_boxes,
        "n_images_without_boxes": n_empty, "n_images_without_label_file": n_missing_label,
        "boxes_per_image": (n_boxes / n_images) if n_images else 0.0,
        "class_counts": dict(zip(info.names, class_counts)),
        "label_line_shapes": totals, "issues": issues, "ok": not issues,
    }
