"""Pipeline roles, group-aware splitting, and materialisation.

Why this module exists
----------------------
The first full run of this project leaked in three ways at once:

  * conformal calibration was fitted on images the detector was trained on;
  * frames of the same video (and Roboflow's augmented copies of one source
    image) sat on both sides of the train/val/test boundary;
  * the split that came with the export was used as-is.

Every stage now reads one of four ROLES, and an image belongs to exactly one:

  fit    detector training
  val    detector model selection / early stopping
  calib  fitting the post-hoc layer (calibrator + conformal thresholds)
  test   every reported number

Images that are near-copies of each other must never straddle roles, so the
unit that gets assigned to a role is a GROUP, not an image:

  * all Roboflow copies of one source image  -> one group
  * a run of consecutive frames of one video -> one group (a "segment")
  * a camera / location id, when the dataset's `group_regex` exposes one

Assignment is a deterministic greedy fill that balances both the number of
images and the number of occluded objects across roles, so the scarce,
occluded objects are not all dealt to one role by chance.

The result is materialised under data/processed/<dataset>/ as plain-bbox YOLO
labels with hard-linked images. Training, occlusion computation, calibration
and evaluation all read that one copy, so every stage sees identical boxes.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from confdet.data.occlusion import occlusion_severity_array
from confdet.data.yolo_dataset import (
    DatasetInfo,
    image_size,
    list_split_images,
    load_dataset_info,
    read_label_file,
    rows_to_yolo_lines,
)
from confdet.utils.paths import long_path

ROLES = ("fit", "val", "calib", "test")

# "<original name>_jpg.rf.<hash>"  (Roboflow export file stem)
_ROBOFLOW_STEM = re.compile(r"^(?P<orig>.+?)_(?:jpe?g|png|bmp|webp|tiff?)\.rf\.[0-9A-Za-z]+$", re.I)
# "<video name>_mp4-<frame index>"  (Roboflow's naming for frames of an uploaded video)
_VIDEO_FRAME = re.compile(r"^(?P<video>.+?_(?:mp4|avi|mov|mkv|m4v|wmv|mpg|mpeg))[-_](?P<frame>\d+)$", re.I)


def source_stem(filename: str) -> str:
    """File stem with Roboflow's '_jpg.rf.<hash>' suffix removed."""
    stem = Path(filename).stem
    m = _ROBOFLOW_STEM.match(stem)
    return m.group("orig") if m else stem


def group_key(filename: str, group_regex: str | None = None, video_segment_frames: int = 30) -> str:
    """Leakage group of an image.

    Precedence: (1) `group_regex` if it matches the source stem -- its named
    group "group", else its first capture group, else the whole match;
    (2) video-frame naming -> "<video>#seg<k>", k = frame // video_segment_frames;
    (3) the source stem itself (so Roboflow copies of one image share a group).
    """
    stem = source_stem(filename)
    if group_regex:
        m = re.search(group_regex, stem)
        if m:
            if "group" in m.groupdict() and m.group("group") is not None:
                return f"g:{m.group('group')}"
            if m.groups():
                return f"g:{m.group(1)}"
            return f"g:{m.group(0)}"
    m = _VIDEO_FRAME.match(stem)
    if m:
        seg = int(m.group("frame")) // max(1, int(video_segment_frames))
        return f"video:{m.group('video')}#seg{seg}"
    return f"img:{stem}"


@dataclass
class ImageEntry:
    image_path: Path
    label_path: Path
    source_split: str
    group: str
    rows: list                      # (class, x1, y1, x2, y2) normalised, clipped
    width: int
    height: int
    occlusion: list[float] = field(default_factory=list)
    role: str | None = None

    @property
    def n_objects(self) -> int:
        return len(self.rows)


def load_entries(
    info: DatasetInfo,
    splits: list[str],
    group_regex: str | None = None,
    video_segment_frames: int = 30,
    occlusion_proxy: str = "bottom_y",
) -> list[ImageEntry]:
    """Read every image of the given on-disk splits with its boxes, leakage
    group and per-object occlusion severity.
    """
    entries: list[ImageEntry] = []
    for split in splits:
        for image_path, label_path in list_split_images(info, split):
            w, h = image_size(image_path)
            rows, _ = read_label_file(label_path, info.label_format, strict=True)
            if rows:
                xyxy = np.array([[r[1] * w, r[2] * h, r[3] * w, r[4] * h] for r in rows])
                occ = occlusion_severity_array(xyxy, proxy=occlusion_proxy).tolist()
            else:
                occ = []
            entries.append(ImageEntry(
                image_path=image_path, label_path=label_path, source_split=split,
                group=group_key(image_path.name, group_regex, video_segment_frames),
                rows=rows, width=w, height=h, occlusion=occ,
            ))
    return entries


def assign_roles(
    entries: list[ImageEntry],
    fractions: dict[str, float],
    seed: int = 42,
    occluded_threshold: float = 0.2,
) -> dict[str, int]:
    """Assign every entry a role, in place; whole groups move together.

    Greedy largest-group-first fill. Each group goes to the role that is
    currently furthest below its target, where "below target" is the mean of
    three normalised deficits: images, objects, occluded objects. Returns the
    number of groups per role.
    """
    unknown = set(fractions) - set(ROLES)
    if unknown:
        raise ValueError(f"unknown roles {sorted(unknown)}; valid roles: {ROLES}")
    roles = [r for r in ROLES if fractions.get(r, 0) > 0]
    if not roles:
        raise ValueError("at least one role needs a positive fraction")
    total_frac = sum(fractions[r] for r in roles)
    target = {r: fractions[r] / total_frac for r in roles}

    groups: dict[str, list[ImageEntry]] = {}
    for e in entries:
        groups.setdefault(e.group, []).append(e)

    def weight(members: list[ImageEntry]) -> np.ndarray:
        n_img = len(members)
        n_obj = sum(m.n_objects for m in members)
        n_occ = sum(sum(1 for o in m.occlusion if o >= occluded_threshold) for m in members)
        return np.array([n_img, n_obj, n_occ], dtype=float)

    weights = {g: weight(m) for g, m in groups.items()}
    totals = np.maximum(sum(weights.values(), np.zeros(3)), 1e-9)
    active = totals > 1e-6                       # a dimension with nothing to balance is ignored

    rng = np.random.default_rng(seed)
    order = sorted(groups)                        # deterministic base order
    rng.shuffle(order)
    order.sort(key=lambda g: -weights[g][0])      # stable: largest groups first

    have = {r: np.zeros(3) for r in roles}
    n_groups = {r: 0 for r in roles}
    for g in order:
        def deficit(r: str) -> float:
            d = (target[r] * totals - have[r]) / totals
            return float(d[active].mean())
        best = max(roles, key=lambda r: (deficit(r), -roles.index(r)))
        have[best] += weights[g]
        n_groups[best] += 1
        for e in groups[g]:
            e.role = best
    return n_groups


def _link_or_copy(src: Path, dst: Path) -> None:
    src, dst = long_path(src), long_path(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize(
    entries: list[ImageEntry],
    out_root: str | Path,
    names: list[str],
    source_info: dict | None = None,
) -> dict:
    """Write data/processed/<dataset>/ :

        images/<role>/<file>      hard links (copies if linking is impossible)
        labels/<role>/<stem>.txt  plain "<class> <cx> <cy> <w> <h>"
        data.yaml                 train -> images/fit, val -> images/val,
                                  test -> images/test
        manifest.json             role, source split, group, objects per image

    Existing images/ and labels/ under out_root are replaced. Returns the
    manifest dict.
    """
    out_root = Path(out_root)
    for sub in ("images", "labels"):
        if (out_root / sub).exists():
            shutil.rmtree(out_root / sub)
    out_root.mkdir(parents=True, exist_ok=True)

    used_names: dict[str, set[str]] = {}
    manifest_images = []
    for e in entries:
        if e.role is None:
            raise ValueError(f"{e.image_path} has no role; call assign_roles first")
        img_dir = out_root / "images" / e.role
        lbl_dir = out_root / "labels" / e.role
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        name = e.image_path.name
        taken = used_names.setdefault(e.role, set())
        if Path(name).stem.lower() in taken:      # same stem from two source splits
            name = f"{e.source_split}__{name}"
        taken.add(Path(name).stem.lower())

        dst = img_dir / name
        if os.name == "nt" and len(str(dst.resolve())) >= 255:
            raise OSError(
                f"Windows path too long ({len(str(dst.resolve()))} characters): {dst.resolve()}\n"
                f"Set project.processed_dir in the config to a short folder on the same drive "
                f"(for example D:/confdet_project/confdet_data) and re-run scripts/02_prepare_data.py.")
        _link_or_copy(e.image_path, dst)
        (lbl_dir / (Path(name).stem + ".txt")).write_text(
            "\n".join(rows_to_yolo_lines(e.rows)), encoding="utf-8")
        manifest_images.append({
            "file": name, "role": e.role, "source_split": e.source_split,
            "group": e.group, "n_objects": e.n_objects,
        })

    roles_present = sorted({e.role for e in entries}, key=ROLES.index)
    # data.yaml carries only the keys Ultralytics itself reads. The calib role
    # is addressed by directory (role_images), never through data.yaml.
    data_yaml = {}
    key_for_role = {"fit": "train", "val": "val", "test": "test"}
    for role in roles_present:
        if role in key_for_role:
            data_yaml[key_for_role[role]] = f"images/{role}"
    data_yaml["nc"] = len(names)
    data_yaml["names"] = {i: n for i, n in enumerate(names)}
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    manifest = {"names": names, "roles": roles_present, "source": source_info or {},
                "images": manifest_images}
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def check_no_group_leakage(manifest: dict) -> list[str]:
    """Groups that appear in more than one role (must be empty)."""
    seen: dict[str, set[str]] = {}
    for im in manifest["images"]:
        seen.setdefault(im["group"], set()).add(im["role"])
    return sorted(g for g, roles in seen.items() if len(roles) > 1)


def build_dataset_roles(ds_name: str, ds_cfg: dict, seed: int, occlusion_cfg: dict,
                        processed_root: str | Path = "data/processed") -> dict:
    """End-to-end for one dataset entry of configs/experiment.yaml: read the
    raw export, apply its `roles` rules, materialise, return a summary.
    """
    info = load_dataset_info(ds_cfg["root"], ds_cfg.get("label_format", "bbox"))
    rules = ds_cfg.get("roles")
    if not rules:
        raise ValueError(f"dataset {ds_name!r} has no `roles` rules in the config")

    all_entries: list[ImageEntry] = []
    groups_per_role: dict[str, int] = {}
    used_splits: set[str] = set()
    for i, rule in enumerate(rules):
        splits = [s for s in rule["from"] if s in info.split_dirs]
        missing = [s for s in rule["from"] if s not in info.split_dirs]
        if missing:
            raise ValueError(
                f"{ds_name}: roles rule {i} asks for on-disk split(s) {missing}, but "
                f"{info.root / 'data.yaml'} only defines {list(info.split_dirs)}.")
        overlap = used_splits & set(splits)
        if overlap:
            raise ValueError(f"{ds_name}: on-disk split(s) {sorted(overlap)} used by two roles rules")
        used_splits |= set(splits)
        entries = load_entries(
            info, splits, ds_cfg.get("group_regex"),
            int(ds_cfg.get("video_segment_frames", 30)),
            occlusion_cfg.get("proxy", "bottom_y"),
        )
        counts = assign_roles(entries, rule["to"], seed=seed + i,
                              occluded_threshold=float(occlusion_cfg.get("occluded_threshold", 0.2)))
        for r, c in counts.items():
            groups_per_role[r] = groups_per_role.get(r, 0) + c
        all_entries.extend(entries)

    out_root = Path(processed_root) / ds_name
    manifest = materialize(all_entries, out_root, info.names,
                           source_info={"root": str(info.root), "label_format": info.label_format})
    leaks = check_no_group_leakage(manifest)

    thr = float(occlusion_cfg.get("occluded_threshold", 0.2))
    per_role = {}
    for role in manifest["roles"]:
        es = [e for e in all_entries if e.role == role]
        occ = [o for e in es for o in e.occlusion]
        per_role[role] = {
            "n_images": len(es), "n_objects": len(occ),
            "n_occluded": int(sum(1 for o in occ if o >= thr)),
            "n_groups": groups_per_role.get(role, 0),
        }
    group_sizes: dict[str, int] = {}
    for e in all_entries:
        group_sizes[e.group] = group_sizes.get(e.group, 0) + 1
    sizes = sorted(group_sizes.values(), reverse=True)
    return {
        "dataset": ds_name, "processed_root": str(out_root), "names": info.names,
        "per_role": per_role, "leaking_groups": leaks,
        "n_groups": len(sizes), "largest_groups": sizes[:5],
        "n_video_segment_groups": sum(1 for g in group_sizes if g.startswith("video:")),
        "example_files": [e.image_path.name for e in all_entries[:3]],
    }


def processed_info(ds_name: str, processed_root: str | Path = "data/processed") -> DatasetInfo:
    """DatasetInfo of a materialised dataset (always plain bbox)."""
    root = Path(processed_root) / ds_name
    if not (root / "data.yaml").exists():
        raise FileNotFoundError(
            f"{root / 'data.yaml'} not found -- run scripts/02_prepare_data.py first.")
    return load_dataset_info(root, "bbox")


def role_images(ds_name: str, role: str, processed_root: str | Path = "data/processed") -> list[tuple[Path, Path]]:
    """Sorted [(image_path, label_path)] of one role of a materialised dataset."""
    root = Path(processed_root) / ds_name
    img_dir, lbl_dir = root / "images" / role, root / "labels" / role
    if not img_dir.exists():
        raise FileNotFoundError(
            f"{img_dir} not found -- dataset {ds_name!r} has no {role!r} role. "
            f"Check its `roles` rules in the config and re-run scripts/02_prepare_data.py.")
    from confdet.data.yolo_dataset import IMAGE_EXTENSIONS
    return [(p, lbl_dir / (p.stem + ".txt"))
            for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]


def has_role(ds_name: str, role: str, processed_root: str | Path = "data/processed") -> bool:
    return (Path(processed_root) / ds_name / "images" / role).is_dir()
