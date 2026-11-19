#!/usr/bin/env python3
"""Export the split manifests of every materialised dataset for the Beocat transfer (writes nothing else).

  python scripts/23_export_manifests.py --out D:/confdet_project/transfer/staging/manifests
        [--experiments bmd_dev bmd_v06 pals_rev pals_dev]

For each <processed_dir>/<experiment>/<dataset>/: copies manifest.json, data.yaml (and UA-DETRAC's
annotations_<role>.json) to <out>/<experiment>/<dataset>/ and writes files.sha256, the hash of every
file of the processed folder (data/rebuild.hash_tree: images byte for byte, text with LF line endings,
*.cache excluded). scripts/02d_rebuild_from_manifest.py checks a rebuilt folder against it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.data.rebuild import hash_tree, write_hash_list  # noqa: E402
from confdet.utils.io_utils import site_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiments", nargs="+", default=["bmd_dev", "bmd_v06", "pals_rev", "pals_dev"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    proc, out = Path(site_paths()["processed_dir"]), Path(a.out)
    cache, summary = {}, {}
    for exp in a.experiments:
        for d in sorted(p for p in (proc / exp).iterdir() if (p / "manifest.json").exists()):
            t0 = time.time()
            dst = out / exp / d.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in ["manifest.json", "data.yaml", *[p.name for p in d.glob("annotations_*.json")]]:
                shutil.copy2(d / f, dst / f)
            h = hash_tree(d, a.workers, cache)
            write_hash_list(h, dst / "files.sha256")
            n_img = len(json.loads((d / "manifest.json").read_text(encoding="utf-8"))["images"])
            summary[f"{exp}/{d.name}"] = {"files_hashed": len(h), "manifest_images": n_img}
            print(f"{exp}/{d.name}: {n_img} images in manifest, {len(h)} files hashed in {time.time() - t0:.0f} s", flush=True)
    (out / "export_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
