"""Derived experiments: new materialised folders built from an EXISTING one
without re-splitting it (v0.6).   Used by scripts/02b_derive_experiment.py.

Three operations, none of which reassigns an image that already has a role:

  verify_rule_assignments  re-derive the role of every image of one `roles`
                           rule from the raw data (same code path and seed as
                           scripts/02) and compare it with the existing
                           manifest. A dry run: nothing is written.
  derive_experiment        hard-link every image/label of chosen roles of a
                           source dataset into a new folder, unchanged, then
                             * append the images of a raw split to a role
                               (e.g. BMD-45 train -> fit), and/or
                             * split one existing role into new roles by
                               leakage group (e.g. PALS calib -> fit/val/calib
                               of a reverse-direction experiment).
                           The source folder is never modified.

Hard links need source and destination on the same volume; a copy is made
otherwise (reported in the result).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import yaml

from confdet.data.occlusion import occlusion_severity_array
from confdet.data.splits import ROLES, ImageEntry, assign_roles, group_key, load_entries
from confdet.data.yolo_dataset import image_size, load_dataset_info, read_label_file, rows_to_yolo_lines
from confdet.utils.paths import long_path


def _strip_split_prefix(name: str, split: str) -> str:
    """materialize() prefixes '<split>__' only when two source splits collide."""
    p = f"{split}__"
    return name[len(p):] if name.startswith(p) else name


def verify_rule_assignments(ds_cfg: dict, rule_index: int, seed: int, occlusion_cfg: dict, manifest: dict) -> dict:
    """Recompute the roles of `ds_cfg['roles'][rule_index]` and compare with `manifest`."""
    info = load_dataset_info(ds_cfg["root"], ds_cfg.get("label_format", "bbox"))
    rule = ds_cfg["roles"][rule_index]
    entries = load_entries(info, [s for s in rule["from"] if s in info.split_dirs], ds_cfg.get("group_regex"),
                           int(ds_cfg.get("video_segment_frames", 30)), occlusion_cfg.get("proxy", "bottom_y"))
    assign_roles(entries, rule["to"], seed=seed + rule_index,
                 occluded_threshold=float(occlusion_cfg.get("occluded_threshold", 0.2)))
    recorded = {}
    for im in manifest["images"]:
        if im["source_split"] in rule["from"]:
            recorded[(im["source_split"], _strip_split_prefix(im["file"], im["source_split"]))] = im["role"]
    derived = {(e.source_split, e.image_path.name): e.role for e in entries}
    common = set(recorded) & set(derived)
    mismatched = sorted(k for k in common if recorded[k] != derived[k])
    per_role: dict[str, int] = {}
    for r in derived.values():
        per_role[r] = per_role.get(r, 0) + 1
    return {"rule_index": rule_index, "rule": rule, "seed": seed + rule_index,
            "n_derived": len(derived), "n_in_manifest": len(recorded), "n_compared": len(common),
            "n_identical": len(common) - len(mismatched), "n_mismatched": len(mismatched),
            "mismatched_examples": [list(k) + [recorded[k], derived[k]] for k in mismatched[:10]],
            "only_in_manifest": len(set(recorded) - set(derived)), "only_derived": len(set(derived) - set(recorded)),
            "derived_per_role": per_role, "identical": len(mismatched) == 0 and set(recorded) == set(derived)}


def _link(src: Path, dst: Path) -> bool:
    """Hard link (True) or copy (False)."""
    src, dst = long_path(src), long_path(dst)
    try:
        os.link(src, dst)
        return True
    except OSError:
        shutil.copy2(src, dst)
        return False


def _entries_from_processed(root: Path, manifest: dict, role: str, proxy: str) -> list[ImageEntry]:
    """ImageEntry objects for the images of one role of a materialised dataset."""
    out = []
    for im in manifest["images"]:
        if im["role"] != role:
            continue
        img = root / "images" / role / im["file"]
        lbl = root / "labels" / role / (Path(im["file"]).stem + ".txt")
        rows, _ = read_label_file(lbl, "bbox")
        w, h = image_size(img)
        occ = occlusion_severity_array(
            __import__("numpy").array([[r[1] * w, r[2] * h, r[3] * w, r[4] * h] for r in rows]).reshape(-1, 4),
            proxy=proxy).tolist() if rows else []
        out.append(ImageEntry(img, lbl, im["source_split"], im["group"], rows, w, h, occ, role))
    return out


def derive_experiment(src_root: str | Path, dst_root: str | Path, keep_roles: dict[str, str],
                      append: dict | None = None, subsplit: dict | None = None,
                      occlusion_cfg: dict | None = None) -> dict:
    """Build dst_root from src_root.

    keep_roles: {source role: destination role} copied unchanged (hard links).
    append:     {"root": raw dataset root, "label_format": ..., "split": "train", "role": "fit",
                 "group_regex": ..., "video_segment_frames": ...} -- every image of that raw split
                 is added to `role`.
    subsplit:   {"role": source role, "to": {new role: fraction}, "seed": int} -- the source
                 role's images are divided by leakage group into new roles (assign_roles).
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    occlusion_cfg = occlusion_cfg or {}
    proxy = occlusion_cfg.get("proxy", "bottom_y")
    if dst_root.exists() and any(dst_root.iterdir()):
        raise FileExistsError(f"{dst_root} already exists and is not empty; refusing to overwrite it")
    src_manifest = json.loads((src_root / "manifest.json").read_text(encoding="utf-8"))
    names = src_manifest["names"]
    images, n_linked, n_copied = [], 0, 0

    def put(img_src: Path, rows_or_label, role: str, name: str, meta: dict):
        nonlocal n_linked, n_copied
        (dst_root / "images" / role).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / role).mkdir(parents=True, exist_ok=True)
        if _link(img_src, dst_root / "images" / role / name):
            n_linked += 1
        else:
            n_copied += 1
        lbl_dst = dst_root / "labels" / role / (Path(name).stem + ".txt")
        if isinstance(rows_or_label, Path):
            _link(rows_or_label, lbl_dst)
        else:
            lbl_dst.write_text("\n".join(rows_to_yolo_lines(rows_or_label)), encoding="utf-8")
        images.append({"file": name, "role": role, **meta})

    for im in src_manifest["images"]:
        if im["role"] not in keep_roles:
            continue
        role = keep_roles[im["role"]]
        put(src_root / "images" / im["role"] / im["file"],
            src_root / "labels" / im["role"] / (Path(im["file"]).stem + ".txt"), role, im["file"],
            {"source_split": im["source_split"], "group": im["group"], "n_objects": im["n_objects"],
             "source_role": im["role"]})

    summary = {"source": str(src_root), "kept": dict(keep_roles)}
    if subsplit:
        entries = _entries_from_processed(src_root, src_manifest, subsplit["role"], proxy)
        for e in entries:
            e.role = None
        counts = assign_roles(entries, subsplit["to"], seed=int(subsplit["seed"]),
                              occluded_threshold=float(occlusion_cfg.get("occluded_threshold", 0.2)))
        for e in entries:
            put(e.image_path, e.label_path, e.role, e.image_path.name,
                {"source_split": e.source_split, "group": e.group, "n_objects": e.n_objects,
                 "source_role": subsplit["role"]})
        summary["subsplit"] = {**subsplit, "groups_per_role": counts}

    if append:
        info = load_dataset_info(append["root"], append.get("label_format", "bbox"))
        entries = load_entries(info, [append["split"]], append.get("group_regex"),
                               int(append.get("video_segment_frames", 30)), proxy)
        taken = {Path(im["file"]).stem.lower() for im in images if im["role"] == append["role"]}
        for e in entries:
            name = e.image_path.name
            if Path(name).stem.lower() in taken:
                name = f"{e.source_split}__{name}"
            taken.add(Path(name).stem.lower())
            put(e.image_path, e.rows, append["role"], name,
                {"source_split": e.source_split, "group": e.group, "n_objects": e.n_objects,
                 "source_role": None, "appended": True})
        summary["append"] = {k: v for k, v in append.items()} | {"n_images": len(entries)}

    roles_present = [r for r in ROLES if any(im["role"] == r for im in images)]
    data_yaml = {k: f"images/{r}" for r, k in (("fit", "train"), ("val", "val"), ("test", "test"))
                 if r in roles_present}
    data_yaml.update({"nc": len(names), "names": {i: n for i, n in enumerate(names)}})
    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
                                        encoding="utf-8")
    manifest = {"names": names, "roles": roles_present, "derived_from": summary, "images": images}
    (dst_root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    per_role = {}
    for im in images:
        d = per_role.setdefault(im["role"], {"n_images": 0, "n_objects": 0, "groups": set()})
        d["n_images"] += 1
        d["n_objects"] += im["n_objects"]
        d["groups"].add(im["group"])
    return {"dst": str(dst_root), "n_linked": n_linked, "n_copied": n_copied,
            "per_role": {r: {"n_images": d["n_images"], "n_objects": d["n_objects"], "n_groups": len(d["groups"])}
                         for r, d in per_role.items()},
            "leaking_groups": sorted({im["group"] for im in images if im["role"] in ("fit", "val")} &
                                     {im["group"] for im in images if im["role"] in ("calib", "test")}),
            "summary": summary}
