"""Phase 1b analyses (evaluation/diagnostics.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.data.occlusion import occlusion_severity_array  # noqa: E402
from confdet.evaluation.diagnostics import occluder_diagnostic, paired_recovery, quantile_strata_edges  # noqa: E402
from confdet.evaluation.labeling import build_tables  # noqa: E402
from confdet.evaluation.records import ImageRecord  # noqa: E402
from confdet.evaluation.rq2_recovery import recovery_analysis  # noqa: E402
from confdet.evaluation.simulate import simulate_records  # noqa: E402
from confdet.evaluation.stats import bootstrap_image_weights  # noqa: E402


def test_paired_recovery_matches_the_baseline_comparison_and_is_zero_against_itself():
    t = build_tables(simulate_records(300, seed=5), class_agnostic=True)
    bw = bootstrap_image_weights(t.n_images, 100, seed=2)
    rng = np.random.default_rng(0)
    other = np.clip(t.c_conf + rng.normal(0, 0.05, t.n_cand), 0, 1)
    budgets = [0.1, 0.25, 0.5, 1.0]
    ref = recovery_analysis(t, {"confidence": t.c_conf, "noisy": other}, 0.25, budgets, bw)["policies"]["noisy"]
    p = paired_recovery(t, {"conf": t.c_conf, "noisy": other}, "conf", 0.25, budgets, bw)
    c = p["comparisons"]["noisy minus conf"]
    assert c["all_misses"] == pytest.approx(ref["mean_recovery_delta_vs_confidence"])
    assert c["all_misses_ci"] == pytest.approx(ref["mean_recovery_delta_ci"])
    same = paired_recovery(t, {"a": t.c_conf, "b": t.c_conf.copy()}, "a", 0.25, budgets, bw)["comparisons"]["b minus a"]
    assert same["all_misses"] == pytest.approx(0.0) and same["all_misses_ci"] == pytest.approx((0.0, 0.0))
    with pytest.raises(KeyError):
        paired_recovery(t, {"a": t.c_conf}, "zzz", 0.25, budgets, bw)


def test_quantile_strata_edges_equal_counts_and_merging():
    rng = np.random.default_rng(1)
    u = rng.uniform(0, 1, 4000)
    r = quantile_strata_edges(u, 4, 300)
    assert len(r["edges"]) == 5 and r["edges"][0] == 0.0 and r["edges"][-1] == 1.0
    assert max(r["calib_counts"]) - min(r["calib_counts"]) <= 2
    # o_hat has a point mass at 0: tied quantile edges collapse, small strata merge
    skew = np.r_[np.zeros(3000), rng.uniform(0.0, 1.0, 900)]
    s = quantile_strata_edges(skew, 4, 300)
    assert all(c >= 300 for c in s["calib_counts"]) and s["edges"][0] == 0.0 and s["edges"][-1] == 1.0
    assert len(s["edges"]) - 1 < 4
    tiny = quantile_strata_edges(rng.uniform(0, 1, 500), 4, 300)
    assert len(tiny["edges"]) == 2 or all(c >= 300 for c in tiny["calib_counts"])


def _img(name, cands):
    """Target (100,100,200,200) half covered by a nearer vehicle (100,150,200,250): o_j = 0.5."""
    gt = np.array([[100, 100, 200, 200], [100, 150, 200, 250]], float)
    c = np.array(cands, float).reshape(-1, 5)
    return ImageRecord(name, 1000, 1000, gt, np.zeros(2, int), occlusion_severity_array(gt),
                       c[:, :4], c[:, 4], np.zeros(len(c), int))


def test_occluder_diagnostic_assigns_each_mechanism():
    recs = [
        _img("not_proposed.jpg", [(100, 100, 200, 200, 0.9)]),
        _img("below_cutoff.jpg", [(100, 100, 200, 200, 0.9), (100, 150, 200, 250, 0.1)]),
        _img("duplicate.jpg", [(100, 150, 200, 250, 0.9), (100, 120, 200, 220, 0.8)]),     # IoU 0.54 between them
        _img("detected.jpg", [(100, 170, 200, 250, 0.9), (100, 100, 200, 160, 0.8)]),      # no overlap between them
    ]
    t = build_tables(recs, class_agnostic=True, context_conf=0.25, duplicate_iou=0.5)
    assert t.g_occ[0] == pytest.approx(0.5) and (t.g_ctx[[0, 2, 4, 6]] < 0.30).all()
    d = occluder_diagnostic(recs, t, context_conf=0.25, duplicate_iou=0.5)
    assert d["n_heavily_occluded"] == 4 and d["n_with_low_context"] == 4
    assert d["counts"] == {"occluder_detected": 1, "occluder_detected_but_duplicate": 1, "occluder_below_cutoff": 1,
                           "occluder_not_proposed": 1, "no_nearer_overlapping_vehicle": 0}
    assert sum(d["by_target"]["target_proposed"].values()) == 4
    # a target whose context is high is not counted
    recs2 = [_img("high.jpg", [(100, 150, 200, 250, 0.9), (100, 100, 200, 200, 0.8)])]   # IoU 0.33: not a duplicate
    t2 = build_tables(recs2, class_agnostic=True, context_conf=0.25, duplicate_iou=0.5)
    assert t2.g_ctx[0] >= 0.30
    assert occluder_diagnostic(recs2, t2, 0.25)["n_with_low_context"] == 0


def test_rarity_diagnostic_separates_fine_from_coarse_rarity():
    from confdet.evaluation.diagnostics import rarity_diagnostic
    rng = np.random.default_rng(0)
    names = ["Hatchback", "Sedan", "SUV", "Bus", "Rickshaw"]
    counts = np.array([400, 350, 250, 15, 10])                        # three common car subtypes, two rare (<2%) classes
    f2c = np.array([0, 0, 0, 1, 2])                                   # car subtypes share one coarse class
    n = 3000
    cls = rng.choice(5, n, p=counts / counts.sum())
    # built in: epistemic is HIGH for the car subtypes (they split evidence) and low for the rare classes
    epi = np.where(cls < 3, 0.02, 0.005) + rng.normal(0, 0.002, n)
    alea = rng.uniform(0.05, 0.2, n)
    causes = np.where(rng.random(n) < 0.3, 0, 1)                      # confused / suppressed
    img = rng.integers(0, 300, n)
    d = rarity_diagnostic(epi, alea, cls, np.ones(n, bool), counts, f2c, causes, rng.normal(0, 1, n), img, 300, names,
                          n_boot=100)
    a = d["a_epistemic_all"]
    # reversed on fine classes, interval below zero ...
    assert a["fine_rarity"]["spearman"] < 0 and a["fine_rarity"]["spearman_ci"][1] < 0
    # ... and MORE reversed on coarse classes: merging the subtypes makes "car" the commonest class while it
    # keeps the highest epistemic uncertainty -- so (a) alone cannot refute the split-evidence hypothesis
    assert a["coarse_rarity"]["spearman"] < a["fine_rarity"]["spearman"]
    c = d["c_car_subtypes_vs_rare"]
    assert c["rare_classes"] == ["Bus", "Rickshaw"] and c["difference"] > 0.01 and c["difference_ci"][0] > 0
    assert d["b_epistemic_confusion_misses"]["n"] + d["b_epistemic_visibility_misses"]["n"] == n
    assert [r["class"] for r in d["c_per_class"]][:3] == ["Hatchback", "Sedan", "SUV"]
