"""Dataset loading, group-aware splitting and materialisation, on synthetic
Roboflow-shaped exports written to a temp dir. No real data needed.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.data.splits import (  # noqa: E402
    ROLES, assign_roles, build_dataset_roles, check_no_group_leakage, group_key,
    load_entries, role_images, source_stem,
)
from confdet.data.yolo_dataset import (  # noqa: E402
    LabelFormatError, iter_split, load_dataset_info, parse_label_line, read_label_file,
    validate_dataset,
)


def _write_image(path: Path, w=64, h=48):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (120, 120, 120)).save(path)


def _make_roboflow_export(root: Path, polygon: bool, n_web=30, n_video_frames=90):
    """Roboflow layout: <split>/images, <split>/labels, 'valid' folder, '../' paths.
    Contains web images (each its own group, some with 2 Roboflow copies) and
    frames of one video.
    """
    names = ["AutoRikshaw", "Bike", "Bus"]
    files = []
    for i in range(n_web):
        files.append(f"web-{i}_jpg.rf.{'a' * 8}{i:04d}.jpg")
        if i % 5 == 0:                                   # an augmented copy of the same source
            files.append(f"web-{i}_jpg.rf.{'b' * 8}{i:04d}.jpg")
    for f in range(n_video_frames):
        files.append(f"clip_one_mp4-{f}_jpg.rf.{'c' * 8}{f:04d}.jpg")
    for k, name in enumerate(files):
        split = ("train", "valid", "test")[k % 3]        # deliberately scatter groups across splits
        _write_image(root / split / "images" / name)
        lines = ["0 0.30 0.30 0.20 0.20", "1 0.35 0.45 0.20 0.30"]   # second box overlaps + is nearer
        if polygon and k % 2 == 0:
            lines.append("2 0.60 0.60 0.80 0.60 0.80 0.90 0.60 0.90")
        if k % 7 == 0:
            lines = []                                    # background image
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        (root / split / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(lines))
    (root / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n"
        f"nc: {len(names)}\nnames: {names}\n")
    return files


def test_parse_label_line_shapes():
    c, box, shape = parse_label_line("3 0.5 0.5 0.2 0.4", "bbox")
    assert (c, shape) == (3, "bbox")
    assert all(abs(a - b) < 1e-12 for a, b in zip(box, (0.4, 0.3, 0.6, 0.7)))

    c, box, shape = parse_label_line("1 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6", "polygon")
    assert shape == "polygon" and all(abs(a - b) < 1e-12 for a, b in zip(box, (0.1, 0.2, 0.5, 0.6)))

    # a 5-field box inside a polygon dataset is still a box
    assert parse_label_line("1 0.5 0.5 0.2 0.2", "polygon")[2] == "bbox"


def test_polygon_line_in_bbox_dataset_is_an_error_not_a_guess():
    with pytest.raises(LabelFormatError, match="label_format: polygon"):
        parse_label_line("1 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6", "bbox")
    with pytest.raises(LabelFormatError):
        parse_label_line("1 0.1 0.2 0.5", "polygon")          # 3 numbers: neither shape
    with pytest.raises(LabelFormatError):
        parse_label_line("car 0.5 0.5 0.2 0.2", "bbox")


def test_out_of_bounds_coordinates_are_clipped_and_empty_boxes_dropped(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.05 0.5 0.2 0.2\n0 1.5 1.5 0.1 0.1\n")
    rows, stats = read_label_file(f, "bbox")
    assert len(rows) == 1 and stats["empty_after_clip"] == 1
    c, x1, y1, x2, y2 = rows[0]
    assert x1 == 0.0 and abs(x2 - 0.15) < 1e-12


def test_roboflow_layout_and_validation(tmp_path):
    _make_roboflow_export(tmp_path / "ds", polygon=True)
    info = load_dataset_info(tmp_path / "ds", "polygon")
    assert set(info.split_dirs) == {"train", "val", "test"}
    rep = validate_dataset(info, "val")
    assert rep["ok"] and rep["n_images"] > 0 and rep["label_line_shapes"]["polygon_lines"] > 0
    image_path, boxes = next(iter_split(info, "train"))
    assert image_path.exists()

    # same export declared as bbox: validation reports it, with the fix in the message
    bad = validate_dataset(load_dataset_info(tmp_path / "ds", "bbox"), "train")
    assert not bad["ok"] and "label_format: polygon" in bad["issues"][0]


def test_group_key_rules():
    assert source_stem("web-3_jpg.rf.abc123.jpg") == "web-3"
    assert group_key("web-3_jpg.rf.abc123.jpg") == group_key("web-3_jpg.rf.zzz999.jpg")
    a = group_key("clip_one_mp4-12_jpg.rf.x1.jpg", video_segment_frames=30)
    b = group_key("clip_one_mp4-29_jpg.rf.x2.jpg", video_segment_frames=30)
    c = group_key("clip_one_mp4-30_jpg.rf.x3.jpg", video_segment_frames=30)
    assert a == b and a != c and a.startswith("video:")
    assert group_key("cam0417_20250112_0930.png", group_regex=r"^(?P<group>cam\d+)_") == "g:cam0417"
    assert group_key("plain.png").startswith("img:")


def test_split_keeps_groups_whole_and_hits_fractions(tmp_path):
    _make_roboflow_export(tmp_path / "raw" / "ds", polygon=True)
    cfg = {"root": str(tmp_path / "raw" / "ds"), "label_format": "polygon",
           "video_segment_frames": 10,
           "roles": [{"from": ["train", "val", "test"],
                      "to": {"fit": 0.6, "val": 0.1, "calib": 0.15, "test": 0.15}}]}
    summary = build_dataset_roles("ds", cfg, seed=1, occlusion_cfg={"proxy": "bottom_y"},
                                  processed_root=tmp_path / "processed")
    assert summary["leaking_groups"] == []
    manifest = json.loads((tmp_path / "processed" / "ds" / "manifest.json").read_text())
    assert check_no_group_leakage(manifest) == []
    n = sum(v["n_images"] for v in summary["per_role"].values())
    assert n == len(manifest["images"])
    assert abs(summary["per_role"]["fit"]["n_images"] / n - 0.6) < 0.08
    assert abs(summary["per_role"]["test"]["n_images"] / n - 0.15) < 0.08
    assert summary["n_video_segment_groups"] == 9          # 90 frames / 10 per segment

    # every Roboflow copy of a source image landed in the same role
    role_of = {}
    for im in manifest["images"]:
        role_of.setdefault(source_stem(im["file"]), set()).add(im["role"])
    assert all(len(r) == 1 for r in role_of.values())

    # materialised labels are plain boxes and match the images one to one
    for role in ROLES:
        for img, lbl in role_images("ds", role, tmp_path / "processed"):
            assert lbl.exists()
            for line in lbl.read_text().splitlines():
                assert len(line.split()) == 5
    data_yaml = (tmp_path / "processed" / "ds" / "data.yaml").read_text()
    assert "train: images/fit" in data_yaml and "val: images/val" in data_yaml
    assert "calib" not in data_yaml


def test_two_rules_keep_on_disk_splits_apart(tmp_path):
    _make_roboflow_export(tmp_path / "raw" / "ds", polygon=False)
    cfg = {"root": str(tmp_path / "raw" / "ds"), "label_format": "bbox",
           "roles": [{"from": ["train"], "to": {"fit": 0.9, "val": 0.1}},
                     {"from": ["val", "test"], "to": {"calib": 0.5, "test": 0.5}}]}
    build_dataset_roles("ds", cfg, seed=0, occlusion_cfg={}, processed_root=tmp_path / "p")
    manifest = json.loads((tmp_path / "p" / "ds" / "manifest.json").read_text())
    for im in manifest["images"]:
        if im["source_split"] == "train":
            assert im["role"] in ("fit", "val")
        else:
            assert im["role"] in ("calib", "test")


def test_split_is_deterministic_and_balances_occluded_objects(tmp_path):
    _make_roboflow_export(tmp_path / "ds", polygon=False)
    info = load_dataset_info(tmp_path / "ds", "bbox")
    runs = []
    for _ in range(2):
        entries = load_entries(info, ["train", "val", "test"], video_segment_frames=10)
        assign_roles(entries, {"fit": 0.5, "test": 0.5}, seed=7)
        runs.append([e.role for e in entries])
    assert runs[0] == runs[1]
    occ = {"fit": 0, "test": 0}
    for e in entries:
        occ[e.role] += sum(1 for o in e.occlusion if o >= 0.2)
    assert min(occ.values()) > 0 and abs(occ["fit"] - occ["test"]) <= 0.2 * sum(occ.values())


# ---------------------------------------------------------------------------
# Windows long paths (found in the first real run: labels at 260+ characters
# were read as "no label file" while validation still printed [OK])
# ---------------------------------------------------------------------------
def test_extended_path_form_is_pure_and_correct():
    from confdet.utils.paths import extended_path_str
    assert extended_path_str(r"C:\a\b.jpg") == "\\\\?\\" + r"C:\a\b.jpg"
    assert extended_path_str(r"\\server\share\b.jpg") == "\\\\?\\UNC\\" + r"server\share\b.jpg"
    assert extended_path_str("\\\\?\\" + r"C:\a") == "\\\\?\\" + r"C:\a"      # idempotent


def test_labels_and_images_beyond_260_characters_are_still_read(tmp_path):
    import os
    import shutil
    from confdet.utils.paths import extended_path_str, long_path

    top = tmp_path / "deep"
    deep = top
    while len(str(deep)) < 250:
        deep = deep / ("d" * 40)
    try:
        long_path(deep / "images").mkdir(parents=True, exist_ok=True)
        long_path(deep / "labels").mkdir(parents=True, exist_ok=True)
        stem = "v" * 60
        _write_image(long_path(deep / "images" / (stem + ".jpg")), 64, 48)
        long_path(deep / "labels" / (stem + ".txt")).write_text("0 0.5 0.5 0.2 0.2")
        assert len(str(deep / "labels" / (stem + ".txt"))) >= 260
        rows, _ = read_label_file(deep / "labels" / (stem + ".txt"), "bbox")
        assert len(rows) == 1, "label of a 260+ character path was not read"
        from confdet.data.yolo_dataset import image_size
        assert image_size(deep / "images" / (stem + ".jpg")) == (64, 48)
        if os.name != "nt":
            assert long_path(deep) == deep                      # untouched off Windows
    finally:
        # `top` itself is short, so long_path() would leave it alone and rmtree could not
        # reach the deep children on Windows; force the extended form there.
        shutil.rmtree(extended_path_str(os.path.abspath(top)) if os.name == "nt" else top,
                      ignore_errors=True)
