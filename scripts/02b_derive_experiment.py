#!/usr/bin/env python3
"""Derived experiment folders built from the EXISTING split, never re-splitting it (v0.6).

  python scripts/02b_derive_experiment.py --verify [--config configs/experiment.yaml]
        [--dataset bmd45] [--rule 0]
      Dry run: re-derive the roles of one `roles` rule from the raw data with the
      same code and seed as scripts/02, compare with the existing manifest, write
      outputs/<experiment>/derive/verify_<dataset>_rule<k>.json. Nothing else is written.

  python scripts/02b_derive_experiment.py --build --config configs/experiment_v06.yaml
      Build <processed_dir>/<experiment>/<dataset>/ for every entry of the config's
      `derive:` block (hard links to the source experiment's files, plus appended raw
      images or a group-aware sub-split). Refuses to overwrite an existing folder.
      Writes outputs/<experiment>/derive/build_<dataset>.json.

See src/confdet/data/derive.py.
"""
from __future__ import annotations

import json
from pathlib import Path

from _common import setup

from confdet.data.derive import derive_experiment, verify_rule_assignments
from confdet.utils.io_utils import save_results
from confdet.utils.paths import experiment_dir, processed_root


def main():
    def extra(p):
        p.add_argument("--verify", action="store_true")
        p.add_argument("--build", action="store_true")
        p.add_argument("--dataset", default=None)
        p.add_argument("--rule", type=int, default=0)

    args, cfg, log = setup("derive_experiment", __doc__, extra=extra)
    seed = int(cfg["project"].get("seed", 42))
    out = experiment_dir(cfg) / "derive"
    if args.verify:
        ds = args.dataset or cfg["experiment"]["development"]
        manifest = json.loads((processed_root(cfg) / ds / "manifest.json").read_text(encoding="utf-8"))
        r = verify_rule_assignments(cfg["datasets"][ds], args.rule, seed, cfg["occlusion"], manifest)
        save_results(r, out / f"verify_{ds}_rule{args.rule}.json")
        log.info(f"{ds} rule {args.rule} {r['rule']} (seed {r['seed']}): derived {r['n_derived']}, in manifest "
                 f"{r['n_in_manifest']}, compared {r['n_compared']}, identical {r['n_identical']}, "
                 f"mismatched {r['n_mismatched']} -> {'IDENTICAL' if r['identical'] else 'DIFFERENT'}")
        log.info(f"  per role (derived): {r['derived_per_role']}")
        if not r["identical"]:
            raise SystemExit(f"assignments differ: {r['mismatched_examples']}")
    if args.build:
        plans = cfg.get("derive") or {}
        if not plans:
            raise SystemExit("this config has no `derive:` block")
        for ds, plan in plans.items():
            src = Path(plan["source_processed"])
            dst = processed_root(cfg) / ds
            res = derive_experiment(src, dst, plan["keep_roles"], plan.get("append"), plan.get("subsplit"),
                                    cfg["occlusion"])
            save_results(res, out / f"build_{ds}.json")
            log.info(f"{ds}: {dst} | linked {res['n_linked']}, copied {res['n_copied']} | per role {res['per_role']}"
                     f" | groups shared between training and evaluation roles: {len(res['leaking_groups'])}")
            if res["leaking_groups"]:
                raise SystemExit(f"{ds}: groups leak between training and evaluation roles: "
                                 f"{res['leaking_groups'][:10]}")


if __name__ == "__main__":
    main()
