"""Phase 1 baselines (evaluation/baselines.py): brute-force references for every
threshold search, the guarantees on simulated data, and hand-checked metrics.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.evaluation import baselines as bl  # noqa: E402
from confdet.evaluation.labeling import build_tables  # noqa: E402
from confdet.evaluation.records import ImageRecord  # noqa: E402
from confdet.evaluation.simulate import SimConfig, simulate_records  # noqa: E402
from confdet.evaluation.stats import bootstrap_image_weights  # noqa: E402


def _brute(thresholds, risk_fn, n, alpha, B=1.0):
    """Largest threshold with (sum of losses + B)/(n+1) <= alpha, by trying every threshold."""
    for t in sorted(set(thresholds), reverse=True):
        if (risk_fn(t) + B) / (n + 1) <= alpha + 1e-12:
            return t
    return None


def _objects(rng, n_img=60):
    g_img = np.repeat(np.arange(n_img), rng.integers(0, 5, n_img))
    prop = rng.random(len(g_img)) < 0.9
    score = np.where(prop, np.round(rng.random(len(g_img)), 2), 0.0)
    return g_img, prop, score


def test_crc_fnr_threshold_equals_brute_force():
    rng = np.random.default_rng(0)
    for trial in range(20):
        g_img, prop, score = _objects(rng)
        n = 60
        for alpha in (0.1, 0.3, 0.6):
            fnr_sum = lambda t: bl.per_image_fnr(prop & (score >= t), g_img, n).sum()  # noqa: E731
            want = _brute(list(score[prop]), fnr_sum, n, alpha)
            got = bl.crc_fnr_threshold(score, prop, g_img, n, alpha)
            assert got["threshold"] == want, (trial, alpha)
            if want is not None:
                assert got["bound"] <= alpha + 1e-12


def test_seqcrc_box_count_losses_equal_brute_force_and_cods_definitions():
    rng = np.random.default_rng(1)
    for trial in range(15):
        n = 30
        g_img = np.repeat(np.arange(n), rng.integers(0, 4, n))
        c_img = rng.integers(0, n, 120)
        c_conf = np.round(rng.random(120), 2)
        n_true = np.bincount(g_img, minlength=n)
        for loss in ("box_count_recall", "box_count_threshold"):
            def total(t):
                k = np.bincount(c_img[c_conf >= t], minlength=n)
                if loss == "box_count_recall":   # cods: max(0, (n_true - n_kept)/n_true), 0 without objects
                    return sum(max(0.0, (a - b) / a) if a else 0.0 for a, b in zip(n_true, k))
                return sum(1.0 if b < a else 0.0 for a, b in zip(n_true, k))   # cods: 1 if n_kept < n_true
            for alpha in (0.2, 0.5):
                want = _brute(list(c_conf), total, n, alpha)
                assert bl.seqcrc_confidence(g_img, n, c_img, c_conf, alpha, loss)["threshold"] == want


def test_pac_k_star_and_threshold():
    k = bl.pac_k_star(200, 0.1, 0.05)
    assert binom.cdf(k, 200, 0.1) <= 0.05 < binom.cdf(k + 1, 200, 0.1)
    assert bl.pac_k_star(5, 0.1, 0.05) == -1                     # too few objects for any certificate
    rng = np.random.default_rng(2)
    g_img, prop, score = _objects(rng, 200)
    r = bl.pac_threshold(score, prop, 0.2, 0.05)
    misses = lambda t: int((~(prop & (score >= t))).sum())  # noqa: E731
    cands = sorted(set(score[prop]), reverse=True)
    want = next((t for t in cands if misses(t) <= r["k_star"]), None)
    assert r["threshold"] == want


def test_pac_guarantee_holds_on_repeated_draws():
    rng = np.random.default_rng(3)
    alpha, delta, n, bad = 0.1, 0.05, 400, 0
    for _ in range(300):
        s_cal, s_new = rng.random(n), rng.random(20000)
        t = bl.pac_threshold(s_cal, np.ones(n, bool), alpha, delta)["threshold"]
        bad += (s_new < t).mean() > alpha
    assert bad / 300 <= delta + 0.03


def test_crc_and_seqcrc_guarantees_on_the_simulator():
    calib = build_tables(simulate_records(500, seed=31), class_agnostic=True)
    test = build_tables(simulate_records(500, seed=32), class_agnostic=True)
    bw = bootstrap_image_weights(test.n_images, 100, seed=0)
    for alpha in (0.1, 0.2):
        crc = bl.crc_fnr_threshold(calib.g_score, calib.g_proposed, calib.g_img, calib.n_images, alpha)
        assert crc["achievable"]
        rep = bl.threshold_rule_report(test, crc["threshold"], 0.5, [0, .2, .4, .6, .8, 1], [0, .05, .3, 1], bw)
        assert rep["mean_image_fnr"] <= alpha + 0.03
    # box counting ignores where the boxes are: false alarms count as "recall", so its threshold is never lower
    s = bl.seqcrc_confidence(calib.g_img, calib.n_images, calib.c_img, calib.c_conf, 0.1)
    c = bl.crc_fnr_threshold(calib.g_score, calib.g_proposed, calib.g_img, calib.n_images, 0.1)
    assert s["threshold"] >= c["threshold"]


def _rec(gt, cands, name="a.jpg"):
    g = np.array(gt, float).reshape(-1, 5)
    c = np.array(cands, float).reshape(-1, 6)
    return ImageRecord(name, 1000, 1000, g[:, :4], g[:, 4].astype(int), np.zeros(len(g)),
                       c[:, :4], c[:, 4], c[:, 5].astype(int))


def test_candidate_geometry_by_hand_and_alignment_with_tables():
    r = _rec(gt=[(0, 0, 100, 100, 0), (500, 500, 600, 600, 0)],
             cands=[(0, 0, 100, 100, 0.9, 1),        # matches object 0, IoU 1
                    (0, 0, 100, 50, 0.6, 0),         # duplicate: IoU 0.5 with the 0.9 box; IoU 0.5 with object 0, already taken
                    (520, 500, 620, 600, 0.4, 2),    # IoU 0.667 with object 1
                    (560, 560, 700, 700, 0.3, 0)])   # IoU 0.08 with object 1 only
    g = bl.candidate_geometry([r])
    assert g["tp"].tolist() == [True, False, True, False]
    assert g["match_iou"] == pytest.approx([1.0, 0.0, 2 / 3, 0.0], abs=1e-6)
    assert g["u_higher"] == pytest.approx([0.0, 0.5, 0.0, iou(520, 500, 620, 600, 560, 560, 700, 700)], abs=1e-6)
    # LaECE_0 matching at any IoU > 0: the 0.3 box finds nothing left either (object 1 taken by the 0.4 box)
    assert g["la_target"] == pytest.approx([1.0, 0.0, 2 / 3, 0.0], abs=1e-6)
    t = build_tables(simulate_records(40, seed=9), class_agnostic=True)
    gs = bl.candidate_geometry(simulate_records(40, seed=9))
    assert len(gs["tp"]) == t.n_cand and (gs["tp"] == t.c_tp).all()


def iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    ix, iy = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def test_gilg_bivariate_beta_learns_that_overlapped_boxes_are_duplicates():
    rng = np.random.default_rng(4)
    s, u = rng.uniform(0.05, 0.95, 4000), rng.uniform(0, 1, 4000)
    p_true = 1 / (1 + np.exp(-(3 * np.log(s / (1 - s)) - 4 * u + 1)))
    y = rng.random(4000) < p_true
    m = bl.fit_gilg(s, u, y)
    p = bl.predict_gilg(m, np.full(5, 0.6), np.linspace(0.01, 0.95, 5))
    assert np.all(np.diff(p) < 0)                                         # more overlap -> less likely a TP
    q = bl.predict_gilg(m, np.linspace(0.1, 0.9, 5), np.full(5, 0.01))
    assert np.all(np.diff(q) > 0)                                         # more confident -> more likely


def test_laece0_laace0_lrp_by_hand():
    conf = np.array([0.9, 0.9, 0.5, 0.5])
    target = np.array([0.9, 0.9, 0.0, 1.0])
    assert bl.laece0(conf, target) == pytest.approx(0.0)                  # each bin's mean matches
    assert bl.laece0(np.array([0.8]), np.array([0.6])) == pytest.approx(0.2)
    assert bl.laace0(conf, target, n_bins=2) == pytest.approx(0.0)
    assert bl.lrp(np.ones(5), 0, 0) == pytest.approx(0.0)                  # perfect detector
    assert bl.lrp(np.zeros(0), 0, 7) == pytest.approx(1.0)                 # finds nothing
    assert bl.lrp(np.array([0.75]), 1, 1) == pytest.approx((0.5 + 1 + 1) / 3)


def test_lrp_on_tables_and_optimal_threshold():
    recs = simulate_records(200, seed=12)
    t = build_tables(recs, class_agnostic=True)
    g = bl.candidate_geometry(recs)
    best = bl.olrp_threshold(t, g)
    grid = np.round(np.arange(0.01, 0.991, 0.01), 2)
    vals = [bl.lrp_at(t, g, float(x))["lrp"] for x in grid]
    assert best["olrp"] == pytest.approx(min(vals)) and 0 < best["olrp"] < 1
    a = bl.lrp_at(t, g, 0.5)
    assert a["n_tp"] + a["n_fn"] == t.n_gt


def test_threshold_rule_report_counts():
    recs = simulate_records(100, seed=13)
    t = build_tables(recs, class_agnostic=True)
    bw = bootstrap_image_weights(t.n_images, 50, seed=1)
    r = bl.threshold_rule_report(t, 0.2, 0.5, [0, .2, .4, .6, .8, 1], [0, .05, .3, 1], bw)
    found = t.g_proposed & (t.g_score >= 0.2)
    assert r["coverage"] == pytest.approx(found.mean())
    assert r["flags_per_image"] == pytest.approx(((t.c_conf >= 0.2) & (t.c_conf < 0.5)).sum() / t.n_images)
    assert sum(b["n"] for b in r["by_annotated_occlusion"]) == t.n_gt
    assert sum(b["n"] for b in r["by_occlusion_context"]) == t.n_gt
    assert bl.threshold_rule_report(t, None, 0.5, [0, 1], [0, 1], bw) == {"threshold": None, "achievable": False}


def test_align_to_tables_survives_class_map_and_ignore_filtering():
    # candidate 1 has an unmapped class (dropped); candidate 2 sits on an ignore region (dropped)
    r = ImageRecord("a.jpg", 1000, 1000, np.array([[0, 0, 100, 100], [500, 500, 600, 600]], float),
                    np.array([0, 1]), np.zeros(2),
                    np.array([[0, 0, 100, 100], [200, 200, 300, 300], [500, 500, 600, 600], [700, 0, 800, 90]], float),
                    np.array([0.9, 0.8, 0.7, 0.6]), np.array([0, 2, 0, 0]))
    t = build_tables([r], class_agnostic=True, cand_class_map=np.array([0, 0, -1]), gt_class_map=np.array([0, -1]))
    assert t.c_src.tolist() == [0, 3]
    raw_values = np.array([10.0, 11.0, 12.0, 13.0])
    assert bl.align_to_tables(raw_values, [r], t).tolist() == [10.0, 13.0]
    plain = build_tables(simulate_records(5, seed=1), class_agnostic=True)
    assert all((plain.c_src[plain.c_img == i] == np.arange((plain.c_img == i).sum())).all() for i in range(5))
