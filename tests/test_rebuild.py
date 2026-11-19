"""Rebuilding materialised datasets from transferred manifests (data/rebuild.py, Beocat migration)."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from confdet.data.derive import derive_experiment  # noqa: E402
from confdet.data.rebuild import hash_tree, read_hash_list, rebuild_from_manifest, verify_tree, write_hash_list  # noqa: E402
from confdet.data.splits import build_dataset_roles  # noqa: E402

OCC = {"proxy": "bottom_y", "occluded_threshold": 0.2}


def _raw(root: Path, splits: dict, seed: int = 0, prefix: str = "") -> Path:
    from PIL import Image
    rng = np.random.default_rng(seed)
    for split, n in splits.items():
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            name = f"{prefix}{split}{i}"
            Image.new("RGB", (64, 48), (int(rng.integers(0, 255)), 0, 0)).save(root / "images" / split / f"{name}.png")
            lines = [f"{int(rng.integers(0, 2))} {cx:.4f} {cy:.4f} 0.2000 0.2000"
                     for cx, cy in rng.uniform(0.2, 0.8, (int(rng.integers(1, 4)), 2))]
            (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
    (root / "data.yaml").write_text(yaml.safe_dump({**{s: f"images/{s}" for s in splits}, "nc": 2, "names": ["a", "b"]}))
    return root


def _rebuild_and_verify(manifest_dir: Path, sources: dict, out: Path, expected: dict) -> dict:
    manifest = json.loads((manifest_dir / "manifest.json").read_text())
    r = rebuild_from_manifest(manifest, sources, out, expected=expected)
    for f in ("manifest.json", "data.yaml"):
        shutil.copy2(manifest_dir / f, out / f)
    return {**r, "verify": verify_tree(out, expected)}


def test_rebuild_from_manifest_reproduces_a_split_and_a_derived_dataset_byte_for_byte(tmp_path):
    raw = _raw(tmp_path / "raw", {"val": 40, "train": 30})
    ds_cfg = {"root": str(raw), "label_format": "bbox",
              "roles": [{"from": ["val"], "to": {"fit": 0.6, "val": 0.1, "calib": 0.15, "test": 0.15}}]}
    build_dataset_roles("d", ds_cfg, 42, OCC, tmp_path / "proc")
    src = tmp_path / "proc" / "d"
    # a derived dataset: keep roles + append the raw train split (as experiment_v06 does)
    derived = tmp_path / "proc2" / "d"
    derive_experiment(src, derived, {"fit": "fit", "val": "val", "calib": "calib", "test": "test"},
                      append={"root": str(raw), "label_format": "bbox", "split": "train", "role": "fit"},
                      occlusion_cfg=OCC)
    sources = {"*": {"root": str(raw), "label_format": "bbox"}}
    for folder in (src, derived):
        hashes = hash_tree(folder)
        write_hash_list(hashes, tmp_path / "h.sha256")
        expected = read_hash_list(tmp_path / "h.sha256")
        assert expected == hashes and "manifest.json" in expected
        # dry check first: writes nothing, finds nothing wrong
        man = json.loads((folder / "manifest.json").read_text())
        chk = rebuild_from_manifest(man, sources, tmp_path / "unused", check_only=True, expected=expected, hash_images=True)
        assert chk["n_problems"] == 0 and not (tmp_path / "unused").exists(), chk
        out = tmp_path / "rebuilt" / folder.parent.name
        r = _rebuild_and_verify(folder, sources, out, expected)
        assert r["n_problems"] == 0 and r["verify"]["identical"], r


def test_rebuild_stops_on_any_difference(tmp_path):
    raw = _raw(tmp_path / "raw", {"val": 30})
    ds_cfg = {"root": str(raw), "label_format": "bbox",
              "roles": [{"from": ["val"], "to": {"calib": 0.5, "test": 0.5}}]}
    build_dataset_roles("d", ds_cfg, 42, OCC, tmp_path / "proc")
    src = tmp_path / "proc" / "d"
    expected = hash_tree(src)
    man = json.loads((src / "manifest.json").read_text())
    sources = {"*": {"root": str(raw), "label_format": "bbox"}}
    # the raw data on the new machine differs: one label edited, one image re-encoded, one image missing
    (raw / "labels" / "val" / "val0.txt").write_text("1 0.5000 0.5000 0.2000 0.2000")
    from PIL import Image
    Image.new("RGB", (64, 48), (1, 2, 3)).save(raw / "images" / "val" / "val1.png")
    (raw / "images" / "val" / "val2.png").unlink()
    chk = rebuild_from_manifest(man, sources, tmp_path / "x", check_only=True, expected=expected, hash_images=True)
    c = chk["problem_counts"]
    assert c["not_in_raw"] == 1 and c["image_differs"] == 1 and c["label_differs"] + c["n_objects_differs"] >= 1
    out = tmp_path / "rebuilt"
    r = _rebuild_and_verify(src, sources, out, expected)
    v = r["verify"]
    assert not v["identical"] and len(v["missing"]) == 2 and len(v["mismatched"]) >= 2   # image + label of val2
    # refuses to overwrite
    try:
        rebuild_from_manifest(man, sources, out)
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass
    # an extra file is reported
    (out / "images" / "calib" / "stray.png").write_bytes(b"x")
    assert "images/calib/stray.png" in verify_tree(out, expected)["extra"]
