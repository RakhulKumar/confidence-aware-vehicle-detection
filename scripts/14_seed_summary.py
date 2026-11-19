#!/usr/bin/env python3
"""Seed-to-seed variability of every headline number (see evaluation/seeds.py).

  python scripts/14_seed_summary.py [--family NAME=member1,member2,member3 ...] [--matching MODE]

Default families:  yolo26 = yolo26, yolo26_s1, yolo26_s2
                   rtdetr = rtdetr, rtdetr_s1, rtdetr_s2

Reads the results written by scripts/07 for each member (primary matching
mode unless --matching is given). Writes
outputs/<experiment>/seeds/seed_summary.json; scripts/12 adds it to the digest.
"""
from __future__ import annotations

from _common import setup

from confdet.evaluation.seeds import headline_numbers, seed_summary
from confdet.utils.io_utils import load_results, matching_modes, save_results
from confdet.utils.paths import experiment_dir, results_dir

DEFAULT_FAMILIES = {"yolo26": ["yolo26", "yolo26_s1", "yolo26_s2"],
                    "rtdetr": ["rtdetr", "rtdetr_s1", "rtdetr_s2"]}
FILES = ("summary", "detection", "rq1", "rq2", "rq3", "conformal")


def parse_families(specs: list[str] | None) -> dict[str, list[str]]:
    if not specs:
        return dict(DEFAULT_FAMILIES)
    out = {}
    for s in specs:
        name, _, members = s.partition("=")
        if not name or not members:
            raise SystemExit(f"--family expects NAME=member1,member2,..., got {s!r}")
        out[name] = [m for m in members.split(",") if m]
    return out


def main():
    def extra(p):
        p.add_argument("--family", action="append", default=None, help="NAME=member1,member2,...")
        p.add_argument("--matching", default=None)

    args, cfg, log = setup("seed_summary", __doc__, extra=extra)
    mode = args.matching or matching_modes(cfg)[0]
    out = {"matching": mode, "families": {}}
    for fam, members in parse_families(args.family).items():
        per = {}
        for m in members:
            res = results_dir(cfg, m, mode)
            missing = [k for k in FILES if not (res / f"{k}.json").exists()]
            if missing:
                log.warning(f"{fam}: {m} has no {missing} under {res}; left out")
                continue
            per[m] = headline_numbers({k: load_results(res / f"{k}.json") for k in FILES})
        if len(per) < 2:
            log.warning(f"{fam}: fewer than two members with results; no seed summary")
            continue
        out["families"][fam] = seed_summary(per)
        mt = out["families"][fam]["metrics"]
        log.info(f"===== {fam} ({', '.join(per)}) | {mode} =====")
        for k in ("ap50_class_agnostic", "recall_at_threshold", "false_alarms_per_image",
                  "rq1_odds_ratio_per_0.1_occlusion", "conformal_a0.05.global_confidence.coverage",
                  "rq2.occlusion_aware.delta_all", "rq2.occlusion_aware.delta_occluded",
                  "dece", "ma_ece", "ma_brier", "oce_max"):
            e = mt.get(k, {})
            if e.get("mean") is not None:
                sd = "n/a" if e["sd"] is None else f"{e['sd']:.4f}"
                log.info(f"  {k:<44} mean {e['mean']:.4f} | SD {sd} | n {e['n']}")
    path = experiment_dir(cfg) / "seeds" / "seed_summary.json"
    save_results(out, path)
    log.info(f"Saved {path}")


if __name__ == "__main__":
    main()
