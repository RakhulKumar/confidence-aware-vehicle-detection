#!/usr/bin/env python3
"""Convert UA-DETRAC into a materialised transfer dataset (v0.6 Phase 3b). See data/detrac.py.

  python scripts/02c_convert_detrac.py --config configs/experiment_detrac.yaml [--every 10]

Roles: calib = every `every`-th frame of the TRAIN sequences, test = every `every`-th frame of the
TEST sequences (sequences are disjoint between the two; the sequence is the leakage / bootstrap
group). Images are hard-linked from the raw folder (same drive), never copied. Labels: YOLO bbox
with classes car, van, bus, others, ignore -- "others" and the sequences' ignored regions both map to
null in the class map, so they become ignore regions. Per-vehicle annotations aligned with the label
lines go to annotations_<role>.json: occ_vehicle (headline), occ_naive, occ_nonvehicle, truncation.
Writes <processed_dir>/<experiment>/ua_detrac/{images,labels}/<role>/, data.yaml, manifest.json;
refuses to overwrite an existing folder.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from _common import setup

from confdet.data.detrac import CLASSES, frame_occlusion, parse_sequence, sampled_frames, yolo_label_lines
from confdet.data.yolo_dataset import image_size
from confdet.utils.io_utils import save_results
from confdet.utils.paths import experiment_dir, processed_root

DS = "ua_detrac"


def convert(raw: Path, out: Path, every: int, log) -> dict:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; refusing to overwrite it")
    manifest, summary = [], {}
    for role, split in (("calib", "Train"), ("test", "Test")):
        (out / "images" / role).mkdir(parents=True, exist_ok=True)
        (out / "labels" / role).mkdir(parents=True, exist_ok=True)
        ann, n_img, n_veh, linked, copied = {}, 0, 0, 0, 0
        xmls = sorted((raw / "DETRAC-Annos" / f"DETRAC-{split}-Annotations-XML").glob("*.xml"))
        for x in xmls:
            seq = parse_sequence(x)
            for num in sampled_frames(seq["frames"], every):
                src = raw / "DETRAC-Images" / seq["name"] / f"img{num:05d}.jpg"
                if not src.exists():
                    continue
                name = f"{seq['name']}_img{num:05d}.jpg"
                dst = out / "images" / role / name
                try:
                    os.link(src, dst)
                    linked += 1
                except OSError:
                    raise SystemExit(f"cannot hard-link {src} -> {dst}; raw and processed must be on the same "
                                     f"drive (copies are not allowed for this dataset)")
                w, h = image_size(dst)
                vs = seq["frames"][num]
                lines, kept = yolo_label_lines(vs, seq["ignored"], w, h)
                (out / "labels" / role / f"{Path(name).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                occ = frame_occlusion(vs)
                ann[name] = [occ[i] for i in kept]
                manifest.append({"file": name, "role": role, "source_split": split.lower(), "group": f"seq:{seq['name']}",
                                 "n_objects": len(kept)})
                n_img += 1
                n_veh += len(kept)
        (out / f"annotations_{role}.json").write_text(json.dumps(ann), encoding="utf-8")
        summary[role] = {"sequences": len(xmls), "images": n_img, "vehicles": n_veh, "linked": linked, "copied": copied}
        log.info(f"{role}: {len(xmls)} {split} sequences -> {n_img} frames (every {every}th), {n_veh} vehicles")
    (out / "data.yaml").write_text(yaml.safe_dump({"test": "images/test", "nc": len(CLASSES),
                                                   "names": {i: n for i, n in enumerate(CLASSES)}}, sort_keys=False),
                                   encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"names": list(CLASSES), "roles": ["calib", "test"],
                                                   "source": {"root": str(raw), "every": every},
                                                   "images": manifest}, indent=1), encoding="utf-8")
    return summary


def main():
    args, cfg, log = setup("convert_detrac", __doc__, extra=lambda p: p.add_argument("--every", type=int, default=10))
    raw = Path(cfg["datasets"][DS]["root"])
    out = processed_root(cfg) / DS
    summary = convert(raw, out, args.every, log)
    save_results({"dataset": DS, "out": str(out), **summary}, experiment_dir(cfg) / "derive" / "convert_ua_detrac.json")


if __name__ == "__main__":
    main()
