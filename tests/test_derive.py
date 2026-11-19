"""Derived experiments (scripts/02b): re-deriving existing assignments, and
building new folders from hard links without ever reassigning an image.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.data.derive import derive_experiment, verify_rule_assignments  # noqa: E402
from confdet.data.splits import build_dataset_roles  # noqa: E402

OCC = {"proxy": "bottom_y", "occluded_threshold": 0.2}


def _raw(root: Path, splits: dict[str, int], seed: int = 0, prefix: str = "") -> Path:
    """Raw YOLO dataset: <root>/images/<split>/*.png, labels/<split>/*.txt, data.yaml."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    for split, n in splits.items():
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            name = f"{prefix}{split}{i}"
            Image.new("RGB", (64, 48)).save(root / "images" / split / f"{name}.png")
            lines = []
            for _ in range(int(rng.integers(1, 4))):
                cx, cy = rng.uniform(0.2, 0.8, 2)
                lines.append(f"{int(rng.integers(0, 2))} {cx:.4f} {cy:.4f} 0.2000 0.2000")
            (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
    (root / "data.yaml").write_text(yaml.safe_dump(
        {**{s: f"images/{s}" for s in splits}, "nc": 2, "names": ["a", "b"]}))
    return root


def _source(tmp_path):
    raw = _raw(tmp_path / "raw", {"val": 60})
    ds_cfg = {"root": str(raw), "label_format": "bbox",
              "roles": [{"from": ["val"], "to": {"fit": 0.6, "val": 0.1, "calib": 0.15, "test": 0.15}}]}
    build_dataset_roles("d", ds_cfg, 42, OCC, tmp_path / "proc")
    return ds_cfg, tmp_path / "proc" / "d"


def test_verify_reproduces_existing_assignments_and_catches_a_change(tmp_path):
    ds_cfg, src = _source(tmp_path)
    manifest = json.loads((src / "manifest.json").read_text())
    r = verify_rule_assignments(ds_cfg, 0, 42, OCC, manifest)
    assert r["identical"] and r["n_compared"] == 60 and r["n_identical"] == 60
    assert sum(r["derived_per_role"].values()) == 60
    # a different seed (= what a careless re-run of 02 could do) is caught
    assert not verify_rule_assignments(ds_cfg, 0, 7, OCC, manifest)["identical"]
    # a tampered manifest is caught
    manifest["images"][0]["role"] = "test" if manifest["images"][0]["role"] != "test" else "fit"
    bad = verify_rule_assignments(ds_cfg, 0, 42, OCC, manifest)
    assert not bad["identical"] and bad["n_mismatched"] == 1


def test_append_keeps_every_existing_role_and_adds_a_raw_split_to_fit(tmp_path):
    _, src = _source(tmp_path)
    before = json.loads((src / "manifest.json").read_text())
    listing = sorted(p.relative_to(src).as_posix() for p in src.rglob("*"))
    extra = _raw(tmp_path / "raw_train", {"train": 25}, seed=3, prefix="t")
    res = derive_experiment(src, tmp_path / "new" / "d", {r: r for r in ("fit", "val", "calib", "test")},
                            append={"root": str(extra), "split": "train", "role": "fit"}, occlusion_cfg=OCC)
    m = json.loads((tmp_path / "new" / "d" / "manifest.json").read_text())
    kept = {(im["file"], im["role"]) for im in m["images"] if not im.get("appended")}
    assert kept == {(im["file"], im["role"]) for im in before["images"]}             # nothing reassigned
    assert sum(1 for im in m["images"] if im.get("appended") and im["role"] == "fit") == 25
    assert res["per_role"]["fit"]["n_images"] == sum(im["role"] == "fit" for im in before["images"]) + 25
    assert res["leaking_groups"] == [] and res["n_copied"] == 0
    a = next(im for im in before["images"] if im["role"] == "test")["file"]
    assert os.stat(src / "images" / "test" / a).st_ino == os.stat(tmp_path / "new" / "d" / "images" / "test" / a).st_ino
    assert sorted(p.relative_to(src).as_posix() for p in src.rglob("*")) == listing     # source untouched
    assert yaml.safe_load((tmp_path / "new" / "d" / "data.yaml").read_text())["train"] == "images/fit"
    with pytest.raises(FileExistsError):
        derive_experiment(src, tmp_path / "new" / "d", {"fit": "fit"})


def test_script_02_refuses_a_derived_config(tmp_path):
    import subprocess
    repo = Path(__file__).resolve().parents[1]
    for name in ("experiment_v06.yaml", "experiment_pals_rev.yaml", "experiment_detrac.yaml"):
        proc = subprocess.run([sys.executable, str(repo / "scripts" / "02_prepare_data.py"), "--config",
                               str(repo / "configs" / name), "--check-only"], capture_output=True, text=True, cwd=repo)
        assert proc.returncode != 0 and "re-split it and is refused" in proc.stderr, name


def test_v06_configs_keep_every_existing_role_and_change_only_what_was_decided():
    repo = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((repo / "configs" / "experiment.yaml").read_text("utf-8"))
    v06 = yaml.safe_load((repo / "configs" / "experiment_v06.yaml").read_text("utf-8"))
    assert v06["project"]["seed"] == base["project"]["seed"] == 42
    skip = ("experiment", "project", "derive", "detectors", "headline", "posthoc")
    assert {k: v for k, v in v06.items() if k not in skip} == {k: v for k, v in base.items() if k not in skip}
    # posthoc differs only in the primary alpha (decision D6: 1-alpha = 0.90), which is on the grid
    assert {k: v for k, v in v06["posthoc"].items() if k != "alpha"} == \
        {k: v for k, v in base["posthoc"].items() if k != "alpha"}
    assert v06["posthoc"]["alpha"] == 0.1 and 0.1 in v06["posthoc"]["alpha_grid"]
    # the headline rule was pre-specified before any final-model prediction existed
    h = v06["headline"]
    assert h["rule"] == "mondrian_gilg_iou_aware" and h["primary_alpha"] == 0.1 and h["strata"] == "posthoc.context_edges"
    assert v06["derive"]["bmd45"]["keep_roles"] == {r: r for r in ("fit", "val", "calib", "test")}
    assert v06["derive"]["bmd45"]["append"]["role"] == "fit" and v06["derive"]["bmd45"]["append"]["split"] == "train"
    models = v06["detectors"]["models"]
    batch = {"yolo26m.pt": 4, "rtdetr-l.pt": 3, "yolov8m.pt": 6}
    for name, m in models.items():
        assert m["batch"] == batch[m["weights"]] and m["patience"] == 10 and m["workers"] <= 4, name
    for base_name in ("yolo26", "rtdetr"):
        for s in (1, 2):
            v = dict(models[f"{base_name}_s{s}"])
            assert v.pop("train_args") == {"seed": s} and v == models[base_name]
    assert v06["detectors"]["gpu_share"] == {"memory_fraction": 0.45, "duty_cycle": 0.8}
    assert base["detectors"]["gpu_share"] == {"memory_fraction": 0.45, "duty_cycle": 0.8}
    rev = yaml.safe_load((repo / "configs" / "experiment_pals_rev.yaml").read_text("utf-8"))
    assert rev["derive"]["pals"]["subsplit"] == {"role": "calib", "to": {"fit": 0.7, "val": 0.1, "calib": 0.2},
                                                 "seed": 142}
    assert rev["derive"]["pals"]["keep_roles"] == {"test": "test"}


def test_subsplit_divides_one_role_by_group_and_leaves_others(tmp_path):
    _, src = _source(tmp_path)
    before = json.loads((src / "manifest.json").read_text())
    calib = [im for im in before["images"] if im["role"] == "calib"]
    plan = {"role": "calib", "to": {"fit": 0.7, "val": 0.1, "calib": 0.2}, "seed": 142}
    res = derive_experiment(src, tmp_path / "rev" / "d", {"test": "test"}, subsplit=plan, occlusion_cfg=OCC)
    m = json.loads((tmp_path / "rev" / "d" / "manifest.json").read_text())
    assert {im["file"] for im in m["images"] if im["role"] == "test"} == \
        {im["file"] for im in before["images"] if im["role"] == "test"}
    from_calib = [im for im in m["images"] if im.get("source_role") == "calib"]
    assert {im["file"] for im in from_calib} == {im["file"] for im in calib}
    assert {im["role"] for im in from_calib} <= {"fit", "val", "calib"}
    by_group = {}
    for im in from_calib:
        by_group.setdefault(im["group"], set()).add(im["role"])
    assert all(len(r) == 1 for r in by_group.values())
    assert res["per_role"]["fit"]["n_images"] > res["per_role"]["calib"]["n_images"] >= 1
    res2 = derive_experiment(src, tmp_path / "rev2" / "d", {"test": "test"}, subsplit=plan, occlusion_cfg=OCC)
    assert res2["per_role"] == res["per_role"]                                            # deterministic
