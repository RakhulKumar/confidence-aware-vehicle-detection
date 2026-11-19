#!/usr/bin/env python3
"""Download BMD-45 (Bengaluru Mobility Dataset) from Hugging Face and
convert it into YOLO layout (images/<split>/, labels/<split>/, data.yaml)
under data/raw/bmd45/.

Format confirmed against a real downloaded sample (2026-09-17): the
Hugging Face repo `iisc-aim/BMD-45` has two top-level folders,
`BMD-45-Train/` and `BMD-45-Val/`, each with one `_annotations.coco.json`
(standard COCO format: `images`, `annotations` with `bbox` as
`[x_min, y_min, w, h]` in pixels, `categories`) and images under
`images_000/`, `images_001/`, `images_002/` subfolders. There is NO `Test`
folder in the repo at all -- consistent with the dataset's test-split
annotations being withheld; only `train` and `val` can actually be
downloaded from here. Image basenames are unique within a split across its
`images_NNN/` subfolders (verified: 10194 val images, 10194 unique
basenames), so flattening into a single `images/<split>/` directory is safe
-- `_convert_annotations` still raises loudly if a real collision is ever
found rather than silently overwriting.

The real category list (read from the downloaded JSON's own `categories`
field, not guessed) is 13 classes: Hatchback, Sedan, SUV, MUV, Bus, Truck,
Three-wheeler, Two-wheeler, LCV, Mini-bus, Tempo-traveller, Bicycle, Van
(ids 0-12, identical across train and val). The dataset card's
`dataset_info` schema additionally lists a 14th "Other" label that does not
appear in either split's `categories` array -- names/nc are taken from
`categories` (what's actually annotated), not the schema list.

Usage:
  hf auth login   # one-time; needs a free HF account + token
  python scripts/download_bmd45.py --inspect-only     # see the real file layout first
  python scripts/download_bmd45.py --splits val --out data/raw/bmd45
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("download_bmd45")

HF_REPO = "iisc-aim/BMD-45"


def _require_hf_hub():
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download
    except ImportError as e:
        raise ImportError(
            "This script requires `huggingface_hub`. Install with: "
            "pip install huggingface_hub --break-system-packages"
        ) from e


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink src to dst (same volume, no extra disk usage); fall back to
    a copy if the two paths are on different filesystems/volumes.
    """
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _convert_annotations(raw_dir: Path, out_dir: Path, split: str) -> dict:
    """Convert one split's COCO-JSON annotations + images into YOLO layout. Confirmed against a real
    downloaded BMD-45 sample -- see module docstring. Returns a summary dict
    logged/saved by main() rather than raising, so a missing split is
    visible in the output instead of silently producing an empty dataset.
    """
    # The split name is only in the PARENT FOLDER ("BMD-45-<Split>/"), not
    # the json's own filename ("_annotations.coco.json") -- confirmed
    # against a real download. Match on the folder, case-insensitively (the
    # repo capitalizes it, e.g. "BMD-45-Val").
    candidate_json_files = [
        p for p in raw_dir.rglob("*.json")
        if any(part.lower() == f"bmd-45-{split.lower()}" for part in p.parts)
    ]
    if not candidate_json_files:
        return {
            "split": split, "converted": False,
            "reason": f"No JSON annotation file found under a 'BMD-45-{split.capitalize()}' "
                      f"folder in {raw_dir}. BMD-45's Hugging Face repo only ships Train and "
                      f"Val annotations (Test annotations are withheld) -- this is expected "
                      f"if split == 'test'. Run with --inspect-only to see the actual "
                      f"downloaded file layout for any other split.",
        }

    ann_file = candidate_json_files[0]
    with open(ann_file) as f:
        coco = json.load(f)

    categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
    names_by_id = {c["id"]: c["name"] for c in categories}
    # YOLO class index = position in the id-sorted category list. Identical to the
    # raw id for BMD-45 (ids 0-12) but stays correct if ids ever start at 1 or skip.
    index_by_id = {c["id"]: i for i, c in enumerate(categories)}

    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images_dir = out_dir / "images" / split
    labels_dir = out_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    seen_basenames: dict[str, Path] = {}
    n_written = 0
    n_images_missing = 0
    for image_id, img in images_by_id.items():
        src_image_path = ann_file.parent / img["file_name"]
        basename = Path(img["file_name"]).name
        if basename in seen_basenames and seen_basenames[basename] != src_image_path:
            raise ValueError(
                f"Basename collision flattening BMD-45 {split} images: "
                f"{seen_basenames[basename]} and {src_image_path} both map to "
                f"{basename}. Flattening is unsafe for this split -- preserve the "
                f"images_NNN/ subfolder structure in both images/ and labels/ "
                f"instead of flattening."
            )
        seen_basenames[basename] = src_image_path

        if not src_image_path.exists():
            n_images_missing += 1
            continue
        _link_or_copy(src_image_path, images_dir / basename)

        w, h = img["width"], img["height"]
        lines = []
        for ann in anns_by_image.get(image_id, []):
            x, y, bw, bh = ann["bbox"]  # COCO format: x_min, y_min, width, height
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            if bw <= 0 or bh <= 0 or ann["category_id"] not in index_by_id:
                continue
            lines.append(f"{index_by_id[ann['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        label_path = labels_dir / (Path(basename).stem + ".txt")
        label_path.write_text("\n".join(lines))
        n_written += 1

    return {
        "split": split, "converted": True, "n_images_labeled": n_written,
        "n_images_missing": n_images_missing, "source_json": str(ann_file),
        "names_by_id": {index_by_id[k]: v for k, v in names_by_id.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/bmd45")
    parser.add_argument(
        "--raw-download-dir", default=None,
        help="Where the raw Hugging Face snapshot (images_NNN/*.png + COCO json, before "
             "conversion) is staged. Defaults to a short path under the OS temp dir, NOT "
             "under --out -- huggingface_hub's download cache appends a long hashed "
             "filename per file (e.g. '<64-hex-hash>.incomplete'), and this project's own "
             "path is already deep enough (OneDrive + several nested folders) that "
             "staging there overflows Windows' 260-char MAX_PATH mid-download (hit in "
             "practice: failed 32GB into the val split). Override only if you know your "
             "--out path has headroom.",
    )
    parser.add_argument("--splits", nargs="+", default=["val"],
                         help="Which splits to pull (default: val only, 10,194 images, which "
                              "configs/experiment.yaml divides into fit/val/calib/test. Add 'train' "
                              "(35,792 images, most of the 153 GB) for the stronger arrangement "
                              "described in that config. 'test' is requestable but "
                              "will report converted=False: BMD-45's Hugging Face repo does not "
                              "host a Test folder at all (annotations withheld). Omit 'train' "
                              "unless you specifically need it -- it's the bulk of the dataset.")
    parser.add_argument("--inspect-only", action="store_true",
                         help="Download the repo listing / a small sample and print the real file "
                              "layout, without attempting conversion. Run this FIRST.")
    args = parser.parse_args()

    snapshot_download = _require_hf_hub()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading from Hugging Face: {HF_REPO} ...")
    if args.inspect_only:
        # allow_patterns kept broad but small -- just enough to see structure
        # without pulling the full ~153GB.
        local_dir = snapshot_download(
            repo_id=HF_REPO, repo_type="dataset", local_dir=str(out_dir / "_inspect"),
            allow_patterns=["*.json", "*.md", "*.txt", "README*"],
        )
        logger.info(f"Inspection download complete: {local_dir}")
        logger.info("Inspect the files above, then edit _convert_annotations() in this script "
                     "if the real annotation format differs from the COCO-JSON assumption, and "
                     "re-run without --inspect-only.")
        return

    if args.raw_download_dir:
        raw_dir = Path(args.raw_download_dir)
    else:
        import tempfile
        raw_dir = Path(tempfile.gettempdir()) / "confdet_bmd45_raw_download"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Staging raw download at {raw_dir} (short path, avoids Windows MAX_PATH)")

    # Real repo layout is "BMD-45-<Split>/..." (capitalized) -- confirmed via
    # list_repo_files(), see module docstring.
    patterns = [f"BMD-45-{split.capitalize()}/*" for split in args.splits] + ["README*"]
    local_dir = snapshot_download(
        repo_id=HF_REPO, repo_type="dataset", local_dir=str(raw_dir), allow_patterns=patterns
    )
    logger.info(f"Downloaded to {local_dir}")

    summaries = []
    for split in args.splits:
        logger.info(f"Converting split: {split} ...")
        summary = _convert_annotations(Path(local_dir), out_dir, split)
        summaries.append(summary)
        logger.info(f"  {summary}")

    converted = [s for s in summaries if s["converted"]]
    if not converted:
        logger.error("No split converted -- nothing to write a data.yaml for.")
        sys.exit(1)

    # All converted splits' category lists must agree (same taxonomy) --
    # verified true for train/val in the real download; fail loudly rather
    # than silently picking one if a future split ever disagrees.
    names_by_id_per_split = [s["names_by_id"] for s in converted]
    if any(n != names_by_id_per_split[0] for n in names_by_id_per_split[1:]):
        logger.error(f"Category lists differ across converted splits: {names_by_id_per_split}")
        sys.exit(1)
    names_by_id = names_by_id_per_split[0]
    names = [names_by_id[i] for i in sorted(names_by_id)]

    # data.yaml -- point train/val/test at whichever splits were actually
    # converted; a split not requested/converted is simply omitted rather
    # than pointed at an empty directory.
    import yaml
    data_yaml_path = out_dir / "data.yaml"
    yaml_content = {}
    if data_yaml_path.exists():
        # A previous run may have converted other splits (e.g. val); keep them,
        # so downloading train later does not drop val from data.yaml.
        previous = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
        if previous.get("names") not in (None, names):
            logger.error(f"Existing {data_yaml_path} lists different classes ({previous.get('names')}) "
                         f"than this download ({names}); refusing to mix them.")
            sys.exit(1)
        for key in ("train", "val", "test"):
            if previous.get(key) and (out_dir / previous[key]).exists():
                yaml_content[key] = previous[key]
    yaml_content.update({s["split"]: f"images/{s['split']}" for s in converted})
    yaml_content["nc"] = len(names)
    yaml_content["names"] = names
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(yaml_content, f)

    logger.info(f"data.yaml written to {data_yaml_path} (nc={len(names)}, names={names})")
    logger.info("Next: python scripts/02_prepare_data.py --check-only")


if __name__ == "__main__":
    main()
