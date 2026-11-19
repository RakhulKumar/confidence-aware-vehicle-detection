"""Seed-to-seed variability of the headline numbers.  Used by scripts/14_seed_summary.py.

`headline_numbers` flattens one detector's results files (one matching mode)
into {metric: value}; `seed_summary` gives, per family of training seeds, the
mean, SD (n-1 denominator), min and max of every metric, plus the values.
An SD from three seeds is itself very uncertain; it says how large run-to-run
noise is, not how precisely it is known.
"""
from __future__ import annotations

import numpy as np

ALPHAS = (0.05, 0.10)
RULES = ("global_confidence", "mondrian_confidence", "global_occlusion_aware")
POLICIES = ("mondrian_rank", "occlusion_aware")


def _alpha_row(curve: list[dict], alpha: float) -> dict | None:
    for row in curve or []:
        if abs(float(row["alpha"]) - alpha) < 1e-9:
            return row
    return None


def headline_numbers(res: dict) -> dict[str, float | None]:
    """res: {"summary", "detection", "rq1", "rq2", "rq3", "conformal"} as loaded from
    outputs/<exp>/<detector>/<mode>/results/*.json."""
    out: dict[str, float | None] = {}
    s, det = res["summary"], res["detection"]
    out["operating_threshold"] = s.get("standard_threshold")
    out["ap50_class_agnostic"] = det["class_agnostic"]["ap50"]
    h = s["headline"]["detection"]
    out["recall_at_threshold"] = h["recall_at_standard"]
    out["false_alarms_per_image"] = h["fp_per_image_at_standard"]
    out["recall_at_floor"] = h["recall_at_floor"]
    lm = res["rq1"].get("logistic_miss_model", {})
    out["rq1_odds_ratio_per_0.1_occlusion"] = lm.get("odds_ratio_per_0.1_occlusion")
    out["rq1_partial_spearman"] = res["rq1"].get("partial_spearman_given_size_and_class")
    curve = res["conformal"].get("validity_curve", [])
    for a in ALPHAS:
        row = _alpha_row(curve, a)
        for rule in RULES:
            r = (row or {}).get(rule) or {}
            key = f"conformal_a{a:g}.{rule}"
            out[f"{key}.coverage"] = r.get("coverage")
            out[f"{key}.flags_per_image"] = r.get("flagged_per_image")
            cert = r.get("all_groups_achievable")
            out[f"{key}.certifiable"] = None if cert is None else float(bool(cert))
    pol = res["rq2"].get("policies", {})
    for p in POLICIES:
        e = pol.get(p, {})
        out[f"rq2.{p}.delta_all"] = e.get("mean_recovery_delta_vs_confidence")
        out[f"rq2.{p}.delta_occluded"] = e.get("mean_recovery_occluded_delta_vs_confidence")
    pt = res["rq3"]["at_standard_threshold"]
    for k in ("dece", "ma_ece", "ma_brier", "oce_max", "miss_mass"):
        out[k] = pt.get(k)
    return out


def seed_summary(per_member: dict[str, dict[str, float | None]]) -> dict:
    """per_member: detector name -> headline_numbers(...). Returns per metric
    {values, mean, sd, min, max, n}; metrics missing for a member are skipped
    for that member and n says how many seeds contributed."""
    members = list(per_member)
    keys = []
    for m in members:
        keys += [k for k in per_member[m] if k not in keys]
    metrics = {}
    for k in keys:
        vals = {m: per_member[m].get(k) for m in members}
        v = np.array([x for x in vals.values() if x is not None and np.isfinite(x)], dtype=float)
        metrics[k] = {"values": vals, "n": int(len(v)),
                      "mean": float(v.mean()) if len(v) else None,
                      "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                      "min": float(v.min()) if len(v) else None,
                      "max": float(v.max()) if len(v) else None}
    return {"members": members, "n_members": len(members), "metrics": metrics}
