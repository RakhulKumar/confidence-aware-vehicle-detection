#!/usr/bin/env python3
"""Build deep-ensemble pseudo-detectors from the cached raw candidates of
several training seeds (see evaluation/ensemble.py for the fusion rule).

  python scripts/15_build_ensemble.py [--ensemble NAME=member1,member2,member3 ...] [--iou 0.55]

Default ensembles:  yolo26_ens = yolo26, yolo26_s1, yolo26_s2
                    rtdetr_ens = rtdetr, rtdetr_s1, rtdetr_s2

For every dataset/role whose predictions exist for ALL members, writes
outputs/<experiment>/<NAME>/predictions/<dataset>__<role>.npz in the format
scripts/05 writes, plus outputs/<experiment>/<NAME>/ensemble.json. Scripts
06, 07 and 08 then take --detector / --detectors NAME unchanged. No GPU, no
weights: an ensemble is never trained or re-predicted, only re-fused.
"""
from __future__ import annotations

from _common import setup

from confdet.evaluation.ensemble import ensemble_records
from confdet.evaluation.records import load_records, save_records
from confdet.utils.io_utils import save_results
from confdet.utils.paths import detector_dir, predictions_path

DEFAULT_ENSEMBLES = {"yolo26_ens": ["yolo26", "yolo26_s1", "yolo26_s2"],
                     "rtdetr_ens": ["rtdetr", "rtdetr_s1", "rtdetr_s2"]}


def parse_ensembles(specs: list[str] | None) -> dict[str, list[str]]:
    if not specs:
        return dict(DEFAULT_ENSEMBLES)
    out = {}
    for s in specs:
        name, _, members = s.partition("=")
        if not name or not members:
            raise SystemExit(f"--ensemble expects NAME=member1,member2,..., got {s!r}")
        out[name] = [m for m in members.split(",") if m]
    return out


def build(name: str, members: list[str], iou: float, cfg: dict, log) -> None:
    pred_dirs = [detector_dir(cfg, m) / "predictions" for m in members]
    missing = [m for m, d in zip(members, pred_dirs) if not d.exists()]
    if missing:
        log.error(f"{name}: no cached predictions for {missing}; run scripts/05 for them first. Skipped.")
        return
    stems = sorted(set.intersection(*[{p.stem for p in d.glob("*.npz")} for d in pred_dirs]))
    if not stems:
        log.error(f"{name}: the members share no dataset/role with predictions. Skipped.")
        return
    stats = {"members": members, "iou_threshold": iou, "files": {}}
    for stem in stems:
        dataset, role = stem.split("__")
        rss = [load_records(predictions_path(cfg, m, dataset, role)) for m in members]
        fused, st = ensemble_records(rss, members, iou)
        out = predictions_path(cfg, name, dataset, role)
        save_records(out, fused)
        stats["files"][stem] = st
        log.info(f"{name} {stem}: {st['n_images']} images, {st['n_member_candidates']} member candidates -> "
                 f"{st['n_fused_candidates']} fused | clusters by members {st['clusters_by_number_of_members']}")
    save_results(stats, detector_dir(cfg, name) / "ensemble.json")
    log.info(f"{name}: wrote {len(stems)} prediction files under {detector_dir(cfg, name) / 'predictions'}")


def main():
    def extra(p):
        p.add_argument("--ensemble", action="append", default=None, help="NAME=member1,member2,...")
        p.add_argument("--iou", type=float, default=0.55)

    args, cfg, log = setup("build_ensemble", __doc__, extra=extra)
    for name, members in parse_ensembles(args.ensemble).items():
        build(name, members, args.iou, cfg, log)


if __name__ == "__main__":
    main()
