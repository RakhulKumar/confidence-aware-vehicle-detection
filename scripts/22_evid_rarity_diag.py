#!/usr/bin/env python3
"""Evidential pilot: rarity diagnostic on the CALIB role only (v0.6, 18 Nov). CPU; no model changes;
does not touch the pilot analysis or its gate.

  python scripts/22_evid_rarity_diag.py [--detector yolo26_evid_pilot_b]

Hypothesis under test: epistemic uncertainty falls with class rarity on test because evidence is
per FINE class, so the common, easily confused car subtypes split their evidence. Checks
(evaluation/diagnostics.rarity_diagnostic): (a) epistemic vs fine and vs coarse rarity (coarse =
the common class of BMD-45's class_map); (b) the same within confusion misses and within
visibility misses (rq5 causes at the pilot's own threshold); (c) epistemic for Hatchback / Sedan / SUV
vs rare classes (fit-role share < 2%), plus a per-class table.
Writes outputs/<experiment>/<detector>/class_agnostic/results/evidential_rarity_diag_calib.json.
"""
from __future__ import annotations

import numpy as np
from _common import setup

from confdet.data.splits import processed_info
from confdet.evaluation.diagnostics import rarity_diagnostic
from confdet.evaluation.labeling import class_map_array
from confdet.evaluation.records import load_records
from confdet.evaluation.rq5_miss_causes import classify_misses
from confdet.models.evidential_yolo import beta_uncertainty
from confdet.pipeline import max_f1_threshold, tables_from_records, train_class_counts
from confdet.utils.io_utils import save_results
from confdet.utils.paths import predictions_path, processed_root, results_dir


def main():
    args, cfg, log = setup("evid_rarity_diag", __doc__,
                           extra=lambda p: p.add_argument("--detector", default="yolo26_evid_pilot_b"))
    det, dev, root = args.detector, cfg["experiment"]["development"], processed_root(cfg)
    names = processed_info(dev, root).names
    path = predictions_path(cfg, det, dev, "calib")
    rs = load_records(path)
    ev = np.load(path.with_name(f"{dev}__calib__evidence.npz"))
    tau, _ = max_f1_threshold(tables_from_records(cfg, rs, "class_agnostic", context_conf=0.25))
    agn = tables_from_records(cfg, rs, "class_agnostic", context_conf=tau)
    nat = tables_from_records(cfg, rs, "native", context_conf=tau)
    if len(ev["alpha"]) != agn.n_cand or agn.n_gt != nat.n_gt:
        raise SystemExit("evidence or tables are not aligned")
    u = beta_uncertainty(ev["alpha"], ev["beta"])
    cand = np.clip(agn.g_cand, 0, None)
    f2c = class_map_array(names, cfg["datasets"][dev]["class_map"], cfg["evaluation"]["common_classes"])
    d = rarity_diagnostic(u["epistemic"][cand], u["aleatoric"][cand], nat.g_cls, agn.g_proposed,
                          train_class_counts(dev, len(names), root), f2c, classify_misses(nat, tau), agn.g_area,
                          agn.g_img, agn.n_images, names)
    d.update({"detector": det, "role": "calib", "threshold": tau,
              "note": "calib role only; diagnostic of the reversed epistemic-rarity association seen on test"})
    save_results(d, results_dir(cfg, det, "class_agnostic") / "evidential_rarity_diag_calib.json")
    a = d["a_epistemic_all"]
    log.info(f"(a) epistemic vs fine rarity {a['fine_rarity']['spearman']:.3f} {a['fine_rarity']['spearman_ci']} | "
             f"vs coarse rarity {a['coarse_rarity']['spearman']:.3f} {a['coarse_rarity']['spearman_ci']}")
    for k in ("b_epistemic_confusion_misses", "b_epistemic_visibility_misses"):
        b = d[k]
        log.info(f"(b) {k}: n {b['n']}" + ("" if "note" in b else
                 f" | fine {b['fine_rarity']['spearman']:.3f} {b['fine_rarity']['spearman_ci']} | coarse "
                 f"{b['coarse_rarity']['spearman']:.3f} {b['coarse_rarity']['spearman_ci']}"))
    c = d["c_car_subtypes_vs_rare"]
    log.info(f"(c) car subtypes {c['mean_epistemic_car_subtypes']:.5f} vs rare {c['rare_classes']} "
             f"{c['mean_epistemic_rare']:.5f}: difference {c['difference']:+.5f} {c['difference_ci']}")


if __name__ == "__main__":
    main()
