#!/usr/bin/env python3
"""Rebuild a materialised dataset on a new machine from its TRANSFERRED manifest (Beocat migration).

  python scripts/02d_rebuild_from_manifest.py --config configs/experiment_v06.yaml --bundle <manifests> --dataset bmd45
        Rebuild <processed_dir>/<experiment>/<dataset>/ from the raw export (roles from the manifest,
        never re-split), copy manifest.json / data.yaml from the bundle, then compare every file
        with the bundle's files.sha256. Exit code 1 on a single difference.
  ... --check-only [--hash-images]
        Write nothing: compare regenerated labels (and raw images) with files.sha256.
  ... --verify-only
        Only compare an existing processed folder with files.sha256 (used after 02c for UA-DETRAC).

<bundle>/<experiment>/<dataset>/ holds manifest.json, data.yaml and files.sha256 (scripts/23_export_manifests.py).
Raw sources: the dataset's `root` in the config; a split appended by the config's `derive.<dataset>.append`
comes from that root. Writes outputs/<experiment>/derive/rebuild_<dataset>.json.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from _common import setup

from confdet.data.rebuild import read_hash_list, rebuild_from_manifest, verify_tree
from confdet.utils.io_utils import save_results
from confdet.utils.paths import experiment_dir, processed_root


def sources_for(cfg: dict, ds: str) -> dict:
    base = cfg["datasets"][ds]
    src = {"*": {"root": base["root"], "label_format": base.get("label_format", "bbox"),
                 "group_regex": base.get("group_regex"), "video_segment_frames": base.get("video_segment_frames", 30)}}
    app = ((cfg.get("derive") or {}).get(ds) or {}).get("append")
    if app:
        src[app["split"]] = {"root": app["root"], "label_format": app.get("label_format", "bbox"),
                             "group_regex": app.get("group_regex"), "video_segment_frames": app.get("video_segment_frames", 30)}
    return src


def main():
    def extra(p):
        p.add_argument("--bundle", required=True, help="folder holding <experiment>/<dataset>/{manifest.json,data.yaml,files.sha256}")
        p.add_argument("--dataset", required=True)
        p.add_argument("--check-only", action="store_true")
        p.add_argument("--hash-images", action="store_true")
        p.add_argument("--verify-only", action="store_true")

    args, cfg, log = setup("rebuild_from_manifest", __doc__, extra=extra)
    exp, ds = cfg["experiment"]["name"], args.dataset
    bdir = Path(args.bundle) / exp / ds
    expected = read_hash_list(bdir / "files.sha256")
    out = processed_root(cfg) / ds
    report = {"experiment": exp, "dataset": ds, "bundle": str(bdir), "processed": str(out)}
    if not args.verify_only:
        manifest = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
        r = rebuild_from_manifest(manifest, sources_for(cfg, ds), out, check_only=args.check_only,
                                  expected=expected, hash_images=args.hash_images)
        report["rebuild"] = r
        log.info(f"{exp}/{ds}: {'checked' if args.check_only else 'rebuilt'} {r.get('n_checked', r.get('n_rebuilt'))} "
                 f"of {r['n_manifest']} images; problems {r['problem_counts']}")
        if not args.check_only:
            for f in ("manifest.json", "data.yaml"):
                shutil.copy2(bdir / f, out / f)
    ok = report.get("rebuild", {}).get("n_problems", 0) == 0
    if args.verify_only and (out / "manifest.json").exists():
        # A converter run on this machine (02c for UA-DETRAC) writes the local raw path into
        # manifest.json -> source.root. It is provenance only (nothing reads it). Everything else must
        # be identical; then the transferred manifest is put in place, as for the rebuilt datasets.
        local = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        moved = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
        roots = {"local": (local.get("source") or {}).get("root"), "transferred": (moved.get("source") or {}).get("root")}
        for m in (local, moved):
            if isinstance(m.get("source"), dict):
                m["source"] = {k: v for k, v in m["source"].items() if k != "root"}
        if local != moved:
            raise SystemExit(f"{exp}/{ds}: manifest.json differs from the transferred one beyond source.root -- stop")
        report["source_root"] = roots
        log.info(f"{exp}/{ds}: manifest.json identical apart from source.root {roots}; transferred copy put in place")
        shutil.copy2(bdir / "manifest.json", out / "manifest.json")
    if not args.check_only:
        v = verify_tree(out, expected)
        report["verify"] = {k: (v[k][:50] if isinstance(v[k], list) else v[k]) for k in v}
        report["verify_counts"] = {k: len(v[k]) for k in ("missing", "extra", "mismatched")}
        log.info(f"{exp}/{ds}: verify against files.sha256: expected {v['n_expected']}, found {v['n_found']}, "
                 f"{report['verify_counts']} -> {'IDENTICAL' if v['identical'] else 'DIFFERENT'}")
        ok = ok and v["identical"]
    report["identical"] = ok
    tag = "check" if args.check_only else ("verify" if args.verify_only else "rebuild")
    save_results(report, experiment_dir(cfg) / "derive" / f"{tag}_{ds}.json")
    if not ok:
        raise SystemExit(f"{exp}/{ds}: NOT identical to the transferred manifest / hash list -- stop. "
                         f"See {experiment_dir(cfg) / 'derive' / f'{tag}_{ds}.json'}")


if __name__ == "__main__":
    main()
