"""Miss-aware conformal thresholds and the occlusion-aware score, checked
against synthetic data with known answers. No detector needed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.evaluation.stats import logistic_irls, sigmoid  # noqa: E402
from confdet.uq.calibrator import fit_calibrator  # noqa: E402
from confdet.uq.conformal import (  # noqa: E402
    NEG_INF, ConformalThresholds, certifiability_by_group, coverage, fit_thresholds, keep_mask, merge_small_strata,
    miss_aware_quantile, rank_transform,
)


def _population(rng, n, p_unproposed=0.03):
    proposed = rng.random(n) > p_unproposed
    return rng.beta(5, 2, n), proposed


def test_quantile_by_hand():
    # n=9, alpha=0.2 -> k = floor(0.2*10) = 2 -> second smallest score
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    g = miss_aware_quantile(scores, np.ones(9, bool), 0.2)
    assert g.achievable and g.threshold == pytest.approx(0.2)
    # one object was never proposed: it occupies the lowest rank, threshold moves down a rank
    prop = np.ones(9, bool); prop[0] = False          # the 0.9 object is in fact unproposed
    g = miss_aware_quantile(scores, prop, 0.2)
    assert g.achievable and g.threshold == pytest.approx(0.1)
    assert g.n_unproposed == 1 and g.min_alpha == pytest.approx(2 / 10)


def test_unreachable_targets_are_reported_not_hidden():
    scores = np.linspace(0.1, 0.9, 20)
    prop = np.ones(20, bool); prop[:5] = False         # 25% of objects have no candidate
    g = miss_aware_quantile(scores, prop, 0.10)
    assert not g.achievable and g.threshold == NEG_INF
    assert g.min_alpha == pytest.approx(6 / 21) and "no candidate" in g.reason
    # too few calibration objects for alpha
    g = miss_aware_quantile(np.array([0.5, 0.6]), np.ones(2, bool), 0.05)
    assert not g.achievable and "calibration objects" in g.reason


def test_marginal_guarantee_holds_including_unproposed_objects():
    rng = np.random.default_rng(0)
    for alpha in (0.05, 0.1, 0.2):
        cov = []
        for _ in range(300):
            # 1% of objects have no candidate at all: they count as misses, and the
            # threshold has to absorb them (the target stays reachable at this rate)
            s, p = _population(rng, 400, p_unproposed=0.01)
            thr = fit_thresholds(s, p, None, alpha)
            assert thr.groups[0].achievable
            st, pt = _population(rng, 400, p_unproposed=0.01)
            cov.append(coverage(st, pt, None, thr)["coverage"])
        mean = float(np.mean(cov))
        assert mean >= 1 - alpha - 0.004, (alpha, mean)        # guarantee
        assert mean <= 1 - alpha + 0.02, (alpha, mean)         # and not vacuous


def test_global_threshold_undercovers_occluded_objects_and_mondrian_repairs_it():
    rng = np.random.default_rng(1)

    def draw(n):
        ctx = np.where(rng.random(n) < 0.25, rng.uniform(0.3, 0.9, n), 0.0)
        s = np.where(ctx > 0, rng.beta(2, 4, n), rng.beta(6, 2, n))   # occluded -> lower scores
        return s, np.ones(n, bool), ctx

    alpha, edges = 0.1, [0.0, 0.05, 0.5, 1.0]
    g_occ, m_occ, m_all = [], [], []
    for _ in range(60):
        s, p, c = draw(4000)
        glob = fit_thresholds(s, p, None, alpha)
        mond = fit_thresholds(s, p, c, alpha, context_edges=edges, min_group_size=100)
        st, pt, ct = draw(4000)
        occ = ct > 0
        g_occ.append(coverage(st[occ], pt[occ], None, glob)["coverage"])
        m_occ.append(coverage(st[occ], pt[occ], ct[occ], mond)["coverage"])
        m_all.append(coverage(st, pt, ct, mond)["coverage"])
    assert np.mean(g_occ) < 1 - alpha - 0.10      # global rule quietly fails the occluded stratum
    assert np.mean(m_occ) >= 1 - alpha - 0.01     # conditional rule restores it
    assert np.mean(m_all) >= 1 - alpha - 0.005


def test_small_strata_merge_into_neighbours():
    assert merge_small_strata([500, 300, 40, 10, 5], 100) == [0, 1, 1, 1, 1]
    assert merge_small_strata([500, 300, 200], 100) == [0, 1, 2]
    assert merge_small_strata([20, 300, 200], 100) == [0, 0, 1]
    assert merge_small_strata([5, 5, 5], 100) == [0, 0, 0]
    rng = np.random.default_rng(2)
    n = 1000
    ctx = np.where(rng.random(n) < 0.02, 0.9, 0.0)             # ~20 objects in the top stratum
    thr = fit_thresholds(rng.random(n), np.ones(n, bool), ctx, 0.1,
                         context_edges=[0.0, 0.5, 1.0], min_group_size=100)
    assert len(thr.groups) == 1 and thr.group_of_stratum == [0, 0]


def test_rank_transform_reproduces_the_threshold_rule_at_every_alpha():
    rng = np.random.default_rng(3)
    n = 3000
    ctx = rng.choice([0.0, 0.2, 0.7], n)
    s, p = _population(rng, n, p_unproposed=0.04)
    edges = [0.0, 0.05, 0.5, 1.0]
    cand_s, cand_c = rng.random(5000), rng.choice([0.0, 0.2, 0.7], 5000)
    base = fit_thresholds(s, p, ctx, 0.1, context_edges=edges, min_group_size=50)
    r = rank_transform(cand_s, cand_c, base)
    assert r.min() > 0 and r.max() <= 1
    for alpha in (0.02, 0.05, 0.1, 0.25, 0.5):
        thr = fit_thresholds(s, p, ctx, alpha, context_edges=edges, min_group_size=50)
        assert np.array_equal(keep_mask(cand_s, cand_c, thr), r > alpha), alpha


def test_thresholds_round_trip_through_json():
    import json
    rng = np.random.default_rng(4)
    s, p = _population(rng, 500, p_unproposed=0.3)             # forces an unreachable (-inf) threshold
    thr = fit_thresholds(s, p, rng.random(500), 0.1, context_edges=[0.0, 0.5, 1.0], min_group_size=50)
    assert any(not g.achievable for g in thr.groups)
    back = ConformalThresholds.from_dict(json.loads(json.dumps(thr.to_dict())))
    cs, cc = rng.random(200), rng.random(200)
    assert np.array_equal(keep_mask(cs, cc, thr), keep_mask(cs, cc, back))
    assert np.allclose(rank_transform(cs, cc, thr), rank_transform(cs, cc, back))


def test_logistic_irls_recovers_known_coefficients():
    rng = np.random.default_rng(5)
    n = 40000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < sigmoid(-0.5 + 1.5 * x1 - 0.8 * x2)).astype(float)
    beta = logistic_irls(np.column_stack([np.ones(n), x1, x2]), y)
    assert np.allclose(beta, [-0.5, 1.5, -0.8], atol=0.06)
    # integer weights == repeating rows
    w = rng.integers(0, 3, n).astype(float)
    b_w = logistic_irls(np.column_stack([np.ones(n), x1, x2]), y, weights=w)
    rep = np.repeat(np.arange(n), w.astype(int))
    b_r = logistic_irls(np.column_stack([np.ones(len(rep)), x1[rep], x2[rep]]), y[rep])
    assert np.allclose(b_w, b_r, atol=1e-4)
    # perfectly separable data must not blow up
    b = logistic_irls(np.column_stack([np.ones(100), np.r_[-np.ones(50), np.ones(50)]]),
                      np.r_[np.zeros(50), np.ones(50)])
    assert np.all(np.isfinite(b))


def _candidates(rng, n, informative):
    """Candidates whose true-positive probability depends on context iff `informative`."""
    conf = rng.beta(1.2, 3, n)
    ctx = np.where(rng.random(n) < 0.3, rng.uniform(0.2, 0.9, n), 0.0)
    dup = np.where(rng.random(n) < 0.2, rng.uniform(0.5, 0.9, n), 0.0)
    z = np.log(conf / (1 - conf))
    eta = 0.5 + 1.2 * z + (2.5 * ctx - 3.0 * dup if informative else 0.0)
    tp = rng.random(n) < sigmoid(eta)
    return conf, ctx, dup, tp


def _tp_at_fp_budget(score, tp, budget):
    order = np.argsort(-score)
    fp = np.cumsum(~tp[order]); t = np.cumsum(tp[order])
    k = np.searchsorted(fp, budget, side="right") - 1
    return int(t[k]) if k >= 0 else 0


def test_occlusion_aware_score_beats_confidence_only_when_context_is_informative():
    rng = np.random.default_rng(6)
    conf, ctx, dup, tp = _candidates(rng, 30000, informative=True)
    cal = fit_calibrator(conf, ctx, dup, tp, "occlusion_aware")
    platt = fit_calibrator(conf, ctx, dup, tp, "confidence")
    c2, x2, d2, tp2 = _candidates(rng, 30000, informative=True)
    budget = 600
    ours = _tp_at_fp_budget(cal.predict(c2, x2, d2), tp2, budget)
    base = _tp_at_fp_budget(c2, tp2, budget)
    assert ours > base * 1.10
    # Platt scaling is monotone in confidence: identical ranking, identical curve
    assert _tp_at_fp_budget(platt.predict(c2), tp2, budget) == base


def test_occlusion_aware_score_does_no_harm_when_context_is_uninformative():
    rng = np.random.default_rng(7)
    conf, ctx, dup, tp = _candidates(rng, 30000, informative=False)
    cal = fit_calibrator(conf, ctx, dup, tp, "occlusion_aware")
    c2, x2, d2, tp2 = _candidates(rng, 30000, informative=False)
    ours = _tp_at_fp_budget(cal.predict(c2, x2, d2), tp2, 600)
    base = _tp_at_fp_budget(c2, tp2, 600)
    assert abs(ours - base) <= 0.02 * base


def test_calibrator_refuses_to_fit_on_too_little_data():
    with pytest.raises(ValueError, match="too little calibration data"):
        fit_calibrator(np.full(30, 0.5), np.zeros(30), np.zeros(30), np.ones(30, bool))


def test_certifiability_by_group_names_the_stratum_that_cannot_be_certified():
    # 200 objects in o_hat [0, 0.3): all proposed; 150 in [0.3, 1): 30 never proposed (20%)
    ctx = np.r_[np.full(200, 0.1), np.full(150, 0.6)]
    prop = np.ones(350, bool); prop[200:230] = False
    thr = fit_thresholds(np.linspace(0.1, 0.9, 350), prop, ctx, 0.10, context_edges=[0.0, 0.3, 1.0], min_group_size=100)
    rows = certifiability_by_group(thr)
    assert [(r["o_hat_lo"], r["o_hat_hi"], r["n_calib"], r["n_never_proposed"]) for r in rows] == [(0.0, 0.3, 200, 0), (0.3, 1.0, 150, 30)]
    assert rows[0]["certifiable"] and rows[0]["min_certifiable_alpha"] == pytest.approx(1 / 201)
    assert not rows[1]["certifiable"] and rows[1]["min_certifiable_alpha"] == pytest.approx(31 / 151)
    assert rows[1]["min_certifiable_alpha"] > rows[1]["target_alpha"]
    # a merged group reports the full o_hat range of its strata
    thr = fit_thresholds(np.linspace(0.1, 0.9, 350), prop, ctx, 0.10, context_edges=[0.0, 0.3, 1.0], min_group_size=300)
    assert [(r["o_hat_lo"], r["o_hat_hi"], r["n_calib"]) for r in certifiability_by_group(thr)] == [(0.0, 1.0, 350)]
