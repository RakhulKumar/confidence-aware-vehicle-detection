"""Labelling, calibration metrics and the RQ1-RQ5 analyses, checked on
hand-built cases and on the simulated detector (whose behaviour is known by
construction). No real detector or dataset needed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.calibration.detection_ece import (  # noqa: E402
    Detection, GroundTruth, binned_calibration_error, dece, match_detections_to_ground_truth,
    miss_aware_brier, miss_aware_ece, oce_top1, operating_point,
)
from confdet.data.occlusion import Box  # noqa: E402
from confdet.evaluation.detection_metrics import (  # noqa: E402
    average_precision, detection_summary, usable_for_calibration_analysis,
)
from confdet.evaluation.labeling import build_tables, class_map_array, greedy_match  # noqa: E402
from confdet.evaluation.records import ImageRecord, RecordSet, load_records, save_records  # noqa: E402
from confdet.evaluation.rq1_correlation import occlusion_analysis  # noqa: E402
from confdet.evaluation.rq2_recovery import conformal_report, policy_scores, recovery_analysis  # noqa: E402
from confdet.evaluation.rq3_calibration_gap import (  # noqa: E402
    calibration_analysis, metric_replicates, ranking_comparison, threshold_for_false_alarms,
)
from confdet.evaluation.rq5_miss_causes import CAUSES, classify_misses, miss_cause_analysis  # noqa: E402
from confdet.evaluation.simulate import SimConfig, simulate_records  # noqa: E402
from confdet.evaluation.stats import bootstrap_image_weights  # noqa: E402
from confdet.uq.calibrator import fit_calibrator  # noqa: E402
from confdet.uq.conformal import fit_thresholds  # noqa: E402

EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CTX_EDGES = [0.0, 0.05, 0.3, 1.0]
TAU = 0.25


def _rec(gt, cands, name="a.jpg", w=1000, h=1000):
    """gt: [(x1,y1,x2,y2,cls,occ)], cands: [(x1,y1,x2,y2,conf,cls)]"""
    g = np.array(gt, dtype=float).reshape(-1, 6)
    c = np.array(cands, dtype=float).reshape(-1, 6)
    return ImageRecord(name, w, h, g[:, :4], g[:, 4].astype(int), g[:, 5],
                       c[:, :4], c[:, 4], c[:, 5].astype(int))


# ---------------------------------------------------------------------------
# matching / labelling
# ---------------------------------------------------------------------------
def _reference_match(dets, gts, thr):
    """Independent pure-Python greedy matcher (the original implementation)."""
    def iou(a, b):
        ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0
    order = sorted(range(len(dets)), key=lambda i: -dets[i][4])
    g_by, d_m = [-1] * len(gts), [-1] * len(dets)
    for di in order:
        best, bj = 0.0, -1
        for gj, g in enumerate(gts):
            if g_by[gj] != -1 or g[4] != dets[di][5]:
                continue
            v = iou(dets[di], g)
            if v > best:
                best, bj = v, gj
        if best >= thr and bj >= 0:
            g_by[bj], d_m[di] = di, bj
    return d_m, g_by


def test_greedy_match_agrees_with_reference_and_thresholding_is_a_prefix():
    rng = np.random.default_rng(0)
    for _ in range(40):
        ng, nc = int(rng.integers(0, 12)), int(rng.integers(0, 40))
        gts = [(x, y, x + w, y + h, int(rng.integers(3))) for x, y, w, h in
               zip(rng.uniform(0, 300, ng), rng.uniform(0, 300, ng), rng.uniform(30, 120, ng), rng.uniform(30, 120, ng))]
        dets = []
        for _k in range(nc):
            if gts and rng.random() < 0.7:
                g = gts[int(rng.integers(len(gts)))]
                j = rng.normal(0, 8, 4)
                dets.append((g[0] + j[0], g[1] + j[1], g[2] + j[2], g[3] + j[3], float(rng.random()),
                             g[4] if rng.random() < 0.8 else int(rng.integers(3))))
            else:
                x, y = rng.uniform(0, 300, 2)
                dets.append((x, y, x + 50, y + 50, float(rng.random()), int(rng.integers(3))))
        d = np.array(dets, dtype=float).reshape(-1, 6)
        g = np.array(gts, dtype=float).reshape(-1, 5)
        c2g, g2c = greedy_match(d[:, :4], d[:, 4], d[:, 5].astype(int), g[:, :4], g[:, 4].astype(int), 0.5)
        ref_d, ref_g = _reference_match(dets, gts, 0.5)
        assert c2g.tolist() == ref_d and g2c.tolist() == ref_g
        # matching only the candidates above a threshold == restricting the full matching
        keep = d[:, 4] >= 0.4
        sub_c2g, _ = greedy_match(d[keep, :4], d[keep, 4], d[keep, 5].astype(int), g[:, :4], g[:, 4].astype(int), 0.5)
        assert sub_c2g.tolist() == c2g[keep].tolist()


def test_list_wrapper_matches_detections_to_ground_truth():
    dets = [Detection(Box(0, 0, 10, 10), 0.9, 0), Detection(Box(100, 100, 110, 110), 0.5, 0)]
    gts = [GroundTruth(Box(0, 0, 10, 10), 0)]
    gm, dm = match_detections_to_ground_truth(dets, gts)
    assert gm[0] is dets[0] and dm == [True, False]


def test_tables_scores_objects_and_flags_unproposed_ones():
    r = _rec(gt=[(0, 0, 100, 100, 0, 0.0), (200, 200, 300, 300, 1, 0.7), (500, 500, 600, 600, 0, 0.9)],
             cands=[(0, 0, 100, 100, 0.9, 0), (205, 200, 300, 300, 0.1, 1), (800, 800, 850, 850, 0.3, 0)])
    t = build_tables([r])
    assert t.g_proposed.tolist() == [True, True, False]
    assert np.allclose(t.g_score, [0.9, 0.1, 0.0], atol=1e-6)
    assert t.c_tp.tolist() == [True, True, False]
    assert t.g_loc_cls.tolist() == [0, 1, -1]


def test_class_map_and_ignore_regions():
    # detector classes: 0=Hatchback 1=Sedan 2=Bicycle ; dataset classes: 0=Car 1=Rickshaw 2=Cart
    common = ["car", "three_wheeler"]
    det_map = class_map_array(["Hatchback", "Sedan", "Bicycle"],
                              {"Hatchback": "car", "Sedan": "car", "Bicycle": None}, common)
    gt_map = class_map_array(["Car", "Rickshaw", "Cart"], {"Car": "car", "Rickshaw": "three_wheeler"}, common)
    assert det_map.tolist() == [0, 0, -1] and gt_map.tolist() == [0, 1, -1]
    r = _rec(gt=[(0, 0, 100, 100, 0, 0), (200, 200, 300, 300, 2, 0)],          # a Car and an unmapped Cart
             cands=[(0, 0, 100, 100, 0.9, 1),                                  # Sedan on the Car -> TP
                    (200, 200, 300, 300, 0.8, 0),                              # on the Cart -> ignored, not a FP
                    (500, 500, 600, 600, 0.7, 2),                              # Bicycle -> unscorable, dropped
                    (700, 700, 800, 800, 0.6, 0)])                             # real false alarm
    t = build_tables([r], cand_class_map=det_map, gt_class_map=gt_map)
    assert t.n_gt == 1 and t.g_proposed.tolist() == [True]
    assert t.n_cand == 2 and t.c_tp.tolist() == [True, False]
    # comparing raw class indices instead (the original RQ4 bug) finds nothing
    raw = build_tables([r])
    assert not raw.g_proposed.any()
    with pytest.raises(ValueError):
        class_map_array(["A"], {"A": "not_in_common"}, common)


def test_records_round_trip(tmp_path):
    recs = simulate_records(5, seed=3)
    save_records(tmp_path / "x.npz", RecordSet(recs, {"detector": "sim", "names": ["a", "b"]}))
    back = load_records(tmp_path / "x.npz")
    assert back.meta["detector"] == "sim" and len(back) == 5
    for a, b in zip(recs, back.records):
        assert a.image_id == b.image_id and np.allclose(a.cand_conf, b.cand_conf)
        assert np.allclose(a.gt_xyxy, b.gt_xyxy) and np.array_equal(a.gt_cls, b.gt_cls)


# ---------------------------------------------------------------------------
# calibration metrics
# ---------------------------------------------------------------------------
def test_dece_by_hand():
    # bin [0.8,0.9): conf .85,.85 -> one TP of two: |0.5-0.85| = 0.35 ; bin [0.2,0.3): conf .25 FP: 0.25
    conf, tp = np.array([0.85, 0.85, 0.25]), np.array([1, 0, 0])
    assert dece(conf, tp, 10) == pytest.approx((2 / 3) * 0.35 + (1 / 3) * 0.25)
    assert binned_calibration_error(np.zeros(0), np.zeros(0))[0] == 0.0
    # weights == repetition
    assert dece(conf, tp, 10, weights=np.array([2, 1, 3])) == pytest.approx(
        dece(np.repeat(conf, [2, 1, 3]), np.repeat(tp, [2, 1, 3]), 10))


def test_miss_aware_ece_identity_and_limits():
    conf, tp = np.array([0.9, 0.9, 0.6, 0.6]), np.array([1, 1, 1, 0])
    d = dece(conf, tp)
    m = miss_aware_ece(conf, tp, n_missed=0)
    assert m["ma_ece"] == pytest.approx(d) and m["gap"] == pytest.approx(0.0)
    m = miss_aware_ece(conf, tp, n_missed=4)
    assert m["miss_mass"] == pytest.approx(0.5)
    assert m["ma_ece"] == pytest.approx(0.5 * d + 0.5)
    assert m["gap"] == pytest.approx(m["miss_mass"] * (1 - d))
    assert miss_aware_ece(np.zeros(0), np.zeros(0), n_missed=3)["ma_ece"] == pytest.approx(1.0)
    assert miss_aware_brier(np.array([1.0]), np.array([1]), n_missed=1) == pytest.approx(0.5)


def test_silently_dropping_hard_objects_is_free_under_dece_but_not_under_miss_aware_metrics():
    """The failure mode the project is about, as a regression test. Two
    detectors are identical on 80 easy vehicles. On 20 hard ones, HONEST emits
    weak, well-calibrated boxes (about 30% right at confidence 0.3); SILENT
    emits nothing at all.
    """
    rng = np.random.default_rng(0)
    easy = [(i * 20, 0, i * 20 + 15, 15, 0, 0.0) for i in range(40)]
    hard = [(i * 20, 500, i * 20 + 15, 515, 0, 0.8) for i in range(10)]
    easy_c = [(x1, y1, x2, y2, 0.9, 0) for x1, y1, x2, y2, _, _ in easy[:36]] + \
             [(x1, y1 + 200, x2, y2 + 200, 0.9, 0) for x1, y1, x2, y2, _, _ in easy[36:]]   # 90% right at 0.9
    hard_c = [(x1, y1, x2, y2, 0.3, 0) for x1, y1, x2, y2, _, _ in hard[:3]] + \
             [(x1, y1 + 200, x2, y2 + 200, 0.3, 0) for x1, y1, x2, y2, _, _ in hard[3:]]    # 30% right at 0.3
    honest = build_tables([_rec(easy + hard, easy_c + hard_c)])
    silent = build_tables([_rec(easy + hard, easy_c)])
    h, s = operating_point(honest, TAU), operating_point(silent, TAU)
    assert h["dece"] == pytest.approx(0.0, abs=1e-6) and s["dece"] == pytest.approx(0.0, abs=1e-6)
    assert s["recall"] < h["recall"]
    assert s["ma_ece"] > h["ma_ece"] + 0.02            # D-ECE ties them; MA-ECE does not
    assert s["oce_max"] > h["oce_max"]
    assert s["ma_brier"] > h["ma_brier"]
    # the old design scored (confidence 0, "hit" 0) as perfectly calibrated; that must stay dead
    assert s["gap"] > 0 and s["miss_mass"] == pytest.approx(14 / (40 + 14))


def test_ma_brier_decomposition_is_exact():
    from confdet.calibration.detection_ece import miss_aware_brier_decomposition

    conf = np.array([0.9, 0.6, 0.3, 0.8])
    tp = np.array([True, True, False, False])
    d = miss_aware_brier_decomposition(conf, tp, n_missed=2)
    D = 4 + 2
    assert d["false_alarm"] == pytest.approx((0.3 ** 2 + 0.8 ** 2) / D)
    assert d["matched"] == pytest.approx((0.1 ** 2 + 0.4 ** 2) / D)
    assert d["miss"] == pytest.approx(2 / D)
    assert d["ma_brier"] == pytest.approx(miss_aware_brier(conf, tp, 2))
    assert d["share_false_alarm"] + d["share_matched"] + d["share_miss"] == pytest.approx(1.0)
    w = np.array([1.0, 2.0, 0.0, 1.0])
    assert miss_aware_brier_decomposition(conf, tp, 1.0, w)["ma_brier"] == pytest.approx(
        miss_aware_brier(conf, tp, 1.0, w))
    # wired into every operating point; the sweep's terms always sum to its MA-Brier
    _, _, test, _ = _world(True)
    for t in (0.1, 0.5, 0.9):
        p = operating_point(test, t, with_oce=False)
        dd = p["ma_brier_decomposition"]
        assert dd["false_alarm"] + dd["matched"] + dd["miss"] == pytest.approx(p["ma_brier"])
    # lowering the threshold moves mass from the miss term to the false-alarm term
    lo, hi = operating_point(test, 0.1, with_oce=False), operating_point(test, 0.9, with_oce=False)
    assert lo["ma_brier_decomposition"]["share_miss"] < hi["ma_brier_decomposition"]["share_miss"]


def test_oce_by_hand():
    r = _rec(gt=[(0, 0, 100, 100, 0, 0), (200, 200, 300, 300, 0, 0), (500, 500, 600, 600, 0, 0)],
             cands=[(0, 0, 100, 100, 0.8, 0),        # right class: (1-0.8)^2 = 0.04
                    (200, 200, 300, 300, 0.6, 1)])   # wrong class: 1 + 0.6^2 = 1.36 ; third object: 1
    t = build_tables([r])
    assert oce_top1(t, 0.25, "max") == pytest.approx((0.04 + 1.36 + 1.0) / 3)
    assert oce_top1(t, 0.25, "ens") == pytest.approx((0.04 + 1.36 + 1.0) / 3)
    assert oce_top1(t, 0.7, "max") == pytest.approx((0.04 + 1.0 + 1.0) / 3)    # the 0.6 box is no longer kept
    # OCE never sees a false alarm on empty road; MA-ECE does
    r2 = _rec(gt=[(0, 0, 100, 100, 0, 0)],
              cands=[(0, 0, 100, 100, 0.95, 0)] + [(300 + 60 * i, 300, 350 + 60 * i, 350, 0.95, 0) for i in range(5)])
    t1, t2 = build_tables([_rec([(0, 0, 100, 100, 0, 0)], [(0, 0, 100, 100, 0.95, 0)])]), build_tables([r2])
    assert oce_top1(t2, 0.25) == pytest.approx(oce_top1(t1, 0.25))
    assert operating_point(t2, 0.25)["ma_ece"] > operating_point(t1, 0.25)["ma_ece"] + 0.5


def test_oce_class_agnostic_ignores_the_box_class():
    gt = [(0, 0, 100, 100, 0, 0), (500, 500, 600, 600, 0, 0)]
    right = _rec(gt, [(0, 0, 100, 100, 0.8, 0), (500, 500, 600, 600, 0.6, 0), (505, 505, 600, 600, 0.4, 0)])
    wrong = _rec(gt, [(0, 0, 100, 100, 0.8, 3), (500, 500, 600, 600, 0.6, 2), (505, 505, 600, 600, 0.4, 1)])
    for variant in ("max", "ens"):
        a = oce_top1(build_tables([right], class_agnostic=True), 0.25, variant)
        b = oce_top1(build_tables([wrong], class_agnostic=True), 0.25, variant)
        assert a == pytest.approx(b)
    # max: best-overlapping box; ens: (1 - mean confidence)^2 over overlapping kept boxes
    t = build_tables([wrong], class_agnostic=True)
    assert oce_top1(t, 0.25, "max") == pytest.approx(((1 - 0.8) ** 2 + (1 - 0.6) ** 2) / 2)
    assert oce_top1(t, 0.25, "ens") == pytest.approx(((1 - 0.8) ** 2 + (1 - 0.5) ** 2) / 2)
    # class-aware tables still charge the misnamed box
    assert oce_top1(build_tables([wrong]), 0.25, "max") > oce_top1(build_tables([right]), 0.25, "max") + 0.5


def test_average_precision_extremes():
    assert average_precision(np.array([0.9, 0.8]), np.array([1, 1]), 2) == pytest.approx(1.0)
    assert average_precision(np.array([0.9, 0.8]), np.array([0, 0]), 2) == pytest.approx(0.0)
    assert average_precision(np.zeros(0), np.zeros(0), 3) == 0.0
    ap = average_precision(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 1, 0]), 2)
    assert 0.5 < ap < 1.0


# ---------------------------------------------------------------------------
# RQ analyses on the simulated detector
# ---------------------------------------------------------------------------
_CACHE = {}


def _world(informative=True):
    key = informative
    if key not in _CACHE:
        cfg = SimConfig(context_informative=informative)
        if not informative:
            cfg.p_unproposed_occluded = 0.0
        calib = build_tables(simulate_records(700, seed=11, cfg=cfg))
        test = build_tables(simulate_records(700, seed=12, cfg=cfg))
        _CACHE[key] = (cfg, calib, test, bootstrap_image_weights(test.n_images, 200, seed=5))
    return _CACHE[key]


def test_detection_summary_and_gate():
    _, _, test, _ = _world()
    s = detection_summary(test, TAU, ["a", "b", "c", "d", "e"])
    assert 0.8 < s["recall_at_standard"] < 0.95 and s["recall_at_floor"] > s["recall_at_standard"]
    assert 0.5 < s["ap50"] <= 1.0 and set(s["per_class"]) == {"a", "b", "c", "d", "e"}
    assert usable_for_calibration_analysis(s)[0]
    ok, msg = usable_for_calibration_analysis({"recall_at_floor": 0.004})
    assert not ok and "class mapping" in msg


def test_rq1_recovers_the_known_occlusion_effect():
    _, _, test, bw = _world(True)
    r = occlusion_analysis(test, TAU, EDGES, bw, min_occluded=200)
    assert r["power"]["adequate"]
    assert r["miss_rate_monotone_over_reliable_buckets"]
    assert r["spearman_occlusion_uncertainty"] > 0.2 and r["spearman_ci"][0] > 0
    assert r["partial_spearman_given_size_and_class"] > 0.15 and r["partial_spearman_ci"][0] > 0
    m = r["logistic_miss_model"]
    assert m["fitted"] and m["odds_ratio_per_0.1_occlusion"] > 1.2 and m["odds_ratio_ci"][0] > 1.0
    assert r["spearman_occlusion_vs_deployable_context"] > 0.6
    assert r["buckets"][-1]["miss_rate"] > 5 * r["buckets"][0]["miss_rate"]


def test_rq1_size_tertiles_and_occlusion_by_size_cross_table():
    from confdet.evaluation.rq1_correlation import size_tertile_table

    # 30 images x 3 vehicles: small (20 px), medium (60 px), large (200 px) on a 1000x1000 image.
    # Small ones are always missed; the medium one is missed only when occluded (severity 0.5).
    recs = []
    for i in range(30):
        occ_med = 0.5 if i % 2 else 0.0
        gt = [(10, 10, 30, 30, 0, 0.0), (300, 300, 360, 360, 0, occ_med), (600, 600, 800, 800, 0, 0.0)]
        cands = [(600, 600, 800, 800, 0.9, 0)] + ([] if occ_med else [(300, 300, 360, 360, 0.9, 0)])
        recs.append(_rec(gt, cands, name=f"{i}.jpg"))
    t = build_tables(recs)
    bw = bootstrap_image_weights(t.n_images, 100, seed=0)
    r = size_tertile_table(t, TAU, EDGES, bw, min_bucket_n=5)
    rates = {row["size"]: row["miss_rate"] for row in r["by_size_tertile"]}
    assert [row["n"] for row in r["by_size_tertile"]] == [30, 30, 30]
    assert rates == {"small": 1.0, "medium": pytest.approx(0.5), "large": 0.0}
    lo, hi = r["size_tertile_edges_rel_area"]
    assert 20 * 20 / 1e6 < lo <= 60 * 60 / 1e6 < hi <= 200 * 200 / 1e6
    cell = {(c["bucket"], c["size"]): c for c in r["by_occlusion_and_size"]}
    assert len(r["by_occlusion_and_size"]) == (len(EDGES) - 1) * 3
    assert cell[(0, "medium")]["miss_rate"] == 0.0 and cell[(0, "medium")]["n"] == 15
    assert cell[(2, "medium")]["miss_rate"] == 1.0 and cell[(2, "medium")]["n"] == 15
    assert cell[(4, "large")]["n"] == 0 and cell[(4, "large")]["miss_rate"] is None
    # wired into the RQ1 result
    full = occlusion_analysis(t, TAU, EDGES, bw, min_occluded=1, min_bucket_n=5, n_rank_boot=20, n_logit_boot=20)
    assert full["by_size_tertile"] == r["by_size_tertile"]


def test_rq1_reports_no_effect_when_there_is_none():
    _, _, test, bw = _world(False)
    r = occlusion_analysis(test, TAU, EDGES, bw)
    lo, hi = r["logistic_miss_model"]["odds_ratio_ci"]
    assert lo < 1.0 < hi
    assert abs(r["partial_spearman_given_size_and_class"]) < 0.05


def test_rq1_flags_an_underpowered_split_instead_of_reporting_noise():
    cfg = SimConfig(mean_objects=1.3)                      # sparse scenes, like the original primary dataset
    t = build_tables(simulate_records(220, seed=1, cfg=cfg))
    r = occlusion_analysis(t, TAU, EDGES, bootstrap_image_weights(t.n_images, 50))
    assert not r["power"]["adequate"] and "NOT interpretable" in r["power"]["message"]
    assert sum(not b["reliable"] for b in r["buckets"]) >= 3


def _fit_layer(calib, alpha=0.2):
    half = calib.c_img % 2 == 0                            # calibrator and thresholds use disjoint images
    ghalf = calib.g_img % 2 == 1
    cal = fit_calibrator(calib.c_conf[half], calib.c_ctx[half], calib.c_dup[half], calib.c_tp[half])
    glob = fit_thresholds(calib.g_score[ghalf], calib.g_proposed[ghalf], None, alpha)
    mond = fit_thresholds(calib.g_score[ghalf], calib.g_proposed[ghalf], calib.g_ctx[ghalf], alpha,
                          context_edges=CTX_EDGES, min_group_size=100)
    return cal, glob, mond


def test_rq2_occlusion_aware_policy_wins_at_matched_cost_when_context_is_informative():
    _, calib, test, bw = _world(True)
    cal, _, mond = _fit_layer(calib)
    r = recovery_analysis(test, policy_scores(test, mond, cal), TAU, [0.1, 0.25, 0.5, 1.0], bw)
    assert 0 < r["recoverable_ceiling"] < 1
    base, ours = r["policies"]["confidence"], r["policies"]["occlusion_aware"]
    for row in ours["at_budgets"] + base["at_budgets"]:
        assert 0 <= row["recovery"] <= r["recoverable_ceiling"] + 1e-9
    assert ours["mean_recovery_delta_vs_confidence"] > 0.03
    assert ours["prob_better_than_confidence"] > 0.95
    assert ours["mean_recovery_occluded_delta_vs_confidence"] > 0.03
    # more budget never recovers less
    rec = [row["recovery"] for row in base["at_budgets"]]
    assert rec == sorted(rec)
    assert "mondrian_rank" in r["policies"]


def test_rq2_reports_no_gain_when_context_is_uninformative():
    _, calib, test, bw = _world(False)
    cal, _, mond = _fit_layer(calib)
    r = recovery_analysis(test, policy_scores(test, mond, cal), TAU, [0.1, 0.25, 0.5, 1.0], bw)
    lo, hi = r["policies"]["occlusion_aware"]["mean_recovery_delta_ci"]
    assert lo <= 0.02 and abs(r["policies"]["occlusion_aware"]["mean_recovery_delta_vs_confidence"]) < 0.05


def test_strict_duplicate_feature_is_disjoint_from_occlusion_context():
    from confdet.uq.calibrator import FEATURE_SETS, _design, strict_duplicate

    dup = np.array([0.0, 0.3, 0.49, 0.5, 0.9])
    assert strict_duplicate(dup, 0.5).tolist() == [0.0, 0.0, 0.0, 0.5, 0.9]
    X = _design(np.full(5, 0.3), np.zeros(5), dup, FEATURE_SETS["full_strict"], 0.5)
    assert X[:, 2].tolist() == [0.0, 0.0, 0.0, 0.5, 0.9]
    # an old calibration.json (no duplicate_iou field) still loads and predicts
    from confdet.uq.calibrator import CandidateCalibrator
    old = {"features": ["z"], "mean": [0.0], "std": [1.0], "coef": [0.0, 1.0], "n_fit": 1, "n_positive": 1,
           "name": "confidence"}
    assert CandidateCalibrator.from_dict(old).predict([0.5])[0] == pytest.approx(0.5)


def test_ablation_cv_on_calib_picks_a_context_set_when_context_is_informative():
    from confdet.evaluation.ablation import cv_select, evaluate_on_test, image_folds

    _, calib, test, bw = _world(True)
    folds = image_folds(calib.image_ids, 5, seed=3)
    assert set(folds) == {0, 1, 2, 3, 4} and min(np.bincount(folds)) > 0.1 * calib.n_images
    assert (image_folds(calib.image_ids[::-1], 5, seed=3) == folds[::-1]).all()     # order-independent
    sel = cv_select(calib, TAU, [0.1, 0.25, 0.5, 1.0], seed=3)
    tab = sel["table"]
    assert all(tab[n]["n_folds_fitted"] == 5 for n in tab)
    assert sel["chosen"] in ("z_ctx", "full", "full_strict")
    assert tab[sel["chosen"]]["cv_mean_recovery"] > tab["z_only"]["cv_mean_recovery"] + 0.02

    rows = np.zeros(calib.n_cand, dtype=bool)
    rows[::2] = True
    res = evaluate_on_test(calib, test, TAU, [0.1, 0.25, 0.5, 1.0], bw, rows)
    # Platt ranks exactly like confidence: no difference at all
    assert abs(res["sets"]["z_only"]["mean_recovery_delta_vs_confidence"]) < 1e-3
    chosen = res["sets"][sel["chosen"]]
    assert chosen["mean_recovery_delta_vs_confidence"] > 0.02 and chosen["mean_recovery_delta_ci"][0] > 0
    assert chosen["calibrator"]["n_fit"] == int(rows.sum())


def test_ablation_finds_no_context_gain_when_context_is_uninformative():
    from confdet.evaluation.ablation import cv_select

    _, calib, _, _ = _world(False)
    tab = cv_select(calib, TAU, [0.1, 0.25, 0.5, 1.0], seed=3)["table"]
    assert tab["z_ctx"]["cv_mean_recovery"] < tab["z_only"]["cv_mean_recovery"] + 0.03


def test_conformal_report_marginal_holds_and_mondrian_repairs_occluded_groups():
    _, calib, test, bw = _world(True)
    cal, glob, mond = _fit_layer(calib, alpha=0.2)
    rep = conformal_report(test, {"global": (glob, test.c_conf), "mondrian": (mond, test.c_conf)},
                           TAU, EDGES, bw)
    assert rep["global"]["coverage"] >= 0.8 - 0.03
    # a single global threshold meets the average while failing heavily occluded vehicles ...
    g_top = rep["global"]["by_occlusion_bucket"][-1]["coverage"]
    assert g_top < 0.8 - 0.15
    # ... the context-conditional rule covers each context group it can certify
    for grp in rep["mondrian"]["by_context_group"]:
        if grp["achievable"]:
            assert grp["coverage"] >= 0.8 - 0.05, grp
        else:
            assert grp["min_certifiable_alpha"] > 0.2 and grp["reason"]
    assert rep["mondrian"]["by_occlusion_bucket"][-1]["coverage"] > g_top
    assert rep["mondrian"]["system_recall"] >= rep["standard_threshold"]["coverage"]
    std = [b["coverage"] for b in rep["standard_threshold"]["by_occlusion_bucket"]]
    assert std[0] > std[-1]


def test_rq3_identity_decomposition_and_threshold_preference():
    _, _, test, bw = _world(True)
    sweep = [round(0.05 * k, 2) for k in range(1, 20)]
    r = calibration_analysis(test, TAU, sweep, 10, EDGES, bw)
    p = r["at_standard_threshold"]
    assert r["identity_check"] and p["ma_ece"] > p["dece"] and p["gap_ci"][0] > 0
    shares = [b["share_of_gap"] for b in r["gap_decomposition_by_occlusion"]]
    assert sum(shares) == pytest.approx(1.0)
    assert sum(b["gap_contribution"] for b in r["gap_decomposition_by_occlusion"]) == pytest.approx(p["gap"])
    top = r["gap_decomposition_by_occlusion"][-1]
    assert top["over_representation"] > 3                 # heavily occluded: few objects, many of the misses
    pref = r["threshold_each_metric_prefers"]
    assert pref["dece"]["threshold"] >= pref["ma_ece"]["threshold"]
    assert pref["dece"]["recall_there"] <= pref["ma_ece"]["recall_there"]


def test_rq3_ranking_can_flip_between_dece_and_miss_aware_metrics():
    # CAUTIOUS: proposes nothing for 35% of vehicles, impeccable on the rest.
    # TALKATIVE: finds nearly everything, mildly over-confident.
    cautious = SimConfig(p_unproposed_base=0.35, p_unproposed_occluded=0.3, background_fp_rate=0.2,
                         duplicate_rate=0.05, rare_confusion=0.0, logit_noise=0.5)
    talkative = SimConfig(p_unproposed_base=0.0, p_unproposed_occluded=0.05, background_fp_rate=3.0,
                          duplicate_rate=0.3, rare_confusion=0.0, base_logit=3.5, occlusion_slope=2.0)
    per = {}
    for name, cfg in (("cautious", cautious), ("talkative", talkative)):
        recs = simulate_records(300, seed=21, cfg=cfg)     # same seed -> same scenes for both
        t = build_tables(recs)
        bw = bootstrap_image_weights(t.n_images, 200, seed=9)
        per[name] = {"point": operating_point(t, TAU), "reps": metric_replicates(t, TAU, 10, bw)}
    r = ranking_comparison(per)
    assert r["rankings"]["ma_ece"][0] == "talkative" and r["rankings"]["oce_max"][0] == "talkative"
    row = r["pairwise"][0]
    assert row["ma_ece"]["separates"] and row["ma_ece"]["better_if_separated"] == "talkative"
    if row["dece"]["separates"] and row["dece"]["better_if_separated"] == "cautious":
        assert r["any_ranking_flip_vs_dece"]
    assert per["cautious"]["point"]["miss_mass"] > per["talkative"]["point"]["miss_mass"] + 0.2
    assert ranking_comparison({"only": per["cautious"]})["note"]


def _fake_entry(point: dict, diffs: dict, rng):
    """Detector entry whose bootstrap replicates sit at point + diff + small noise."""
    reps = {m: point[m] + diffs.get(m, 0.0) + rng.normal(0, 0.001, 400) for m in point}
    return {"point": point, "reps": reps}


def test_ranking_flip_requires_both_intervals_to_exclude_zero():
    rng = np.random.default_rng(0)
    base = {"dece": 0.05, "ma_ece": 0.16, "ma_brier": 0.20, "oce_max": 0.28}
    ref = {"point": base, "reps": {m: np.full(400, v) for m, v in base.items()}}

    # D-ECE difference is pure noise around zero; MA-ECE clearly prefers "x"
    noisy = {"point": dict(base, dece=0.049, ma_ece=0.15),
             "reps": {"dece": 0.05 + rng.normal(0, 0.005, 400), "ma_ece": np.full(400, 0.15) + rng.normal(0, 0.001, 400),
                      "ma_brier": np.full(400, 0.20) + rng.normal(0, 0.001, 400),
                      "oce_max": np.full(400, 0.28) + rng.normal(0, 0.001, 400)}}
    r = ranking_comparison({"x": noisy, "y": ref})
    row = r["pairwise"][0]
    assert not row["dece"]["separates"] and row["ma_ece"]["separates"]
    assert row["ma_ece"]["better_if_separated"] == "x"
    assert not row["flips_vs_dece"]["ma_ece"] and not r["any_ranking_flip_vs_dece"]
    assert r["pairs_dece_cannot_separate"] == [["x", "y"]]
    assert "ma_ece" in row["separated_by"] and "dece" not in row["separated_by"]

    # both separate, opposite signs -> a flip
    flip = _fake_entry(base, {"dece": -0.01, "ma_ece": +0.01, "ma_brier": +0.01, "oce_max": -0.01}, rng)
    r = ranking_comparison({"x": flip, "y": ref})
    row = r["pairwise"][0]
    assert row["flips_vs_dece"] == {"ma_ece": True, "ma_brier": True, "oce_max": False}
    assert r["any_ranking_flip_vs_dece"] and r["pairs_dece_cannot_separate"] == []
    assert ["dece", "ma_ece"] in row["flips_between_metrics"] and ["ma_ece", "ma_brier"] not in row["flips_between_metrics"]

    # both separate, same sign -> no flip
    agree = _fake_entry(base, {m: -0.01 for m in base}, rng)
    assert not ranking_comparison({"x": agree, "y": ref})["any_ranking_flip_vs_dece"]


def test_rq5_separates_visibility_misses_from_confusion_misses():
    cfg, _, test, bw = _world(True)
    cause = classify_misses(test, TAU)
    found = test.g_proposed & (test.g_score >= TAU)
    assert ((cause == -1) == found).all()
    train_counts = np.array(cfg.class_probs) * 10000
    r = miss_cause_analysis(test, TAU, train_counts, EDGES, bw, n_boot=100)
    assert sum(r["cause_counts"].values()) == r["n_missed"]
    assert r["mean_occlusion_by_cause"]["suppressed"] > r["mean_occlusion_by_cause"]["confused"] + 0.2
    assert r["mean_rarity_by_cause"]["confused"] > r["mean_rarity_by_cause"]["suppressed"] + 0.5
    v = r["verdict"]
    assert v["visibility_misses_rise_with_occlusion"] and v["confusion_misses_rise_with_rarity"]
    assert v["causes_separated"]
    with pytest.raises(ValueError):
        classify_misses(build_tables(simulate_records(3, seed=0), class_agnostic=True), TAU)


def test_transfer_summary_reports_flags_and_certifiability_next_to_every_coverage():
    from confdet.evaluation.rq4_cross_dataset import transfer_summary

    def rep(cov, flags, cert, target=0.95):
        return {"standard_threshold": {"coverage": 0.8},
                "global_confidence": {"target_coverage": target, "coverage": cov,
                                      "flagged_per_image": flags, "all_groups_achievable": cert}}
    dev = {"conformal": rep(0.95, 4.0, True),
           "conformal_by_alpha": {"0.05": rep(0.95, 4.0, True), "0.1": rep(0.90, 2.0, True, 0.90)}}
    tr = {"conformal_frozen": rep(0.80, 3.0, True), "conformal_recalibrated": rep(0.80, 9.0, False),
          "conformal_by_alpha": {"0.05": {"frozen": rep(0.80, 3.0, True), "recalibrated": rep(0.80, 9.0, False)},
                                 "0.1": {"frozen": rep(0.78, 1.5, True, 0.90),
                                         "recalibrated": rep(0.91, 2.5, True, 0.90)}},
          "gate": {"usable": True, "message": ""}}
    s = transfer_summary(dev, tr)
    row = s["conformal_coverage"]["global_confidence"]
    assert row["transfer_recalibrated_flags_per_image"] == 9.0 and row["transfer_recalibrated_certifiable"] is False
    assert row["development_flags_per_image"] == 4.0 and row["transfer_frozen_certifiable"] is True
    by = s["conformal_coverage_by_alpha"]
    assert set(by) == {"0.05", "0.1"}
    r10 = by["0.1"]["global_confidence"]
    assert r10["target"] == 0.90 and r10["development"] == 0.90 and r10["transfer_recalibrated"] == 0.91
    assert r10["transfer_recalibrated_certifiable"] is True and r10["recalibration_restores"]
    assert "conformal_coverage_by_alpha" not in transfer_summary(dev, {k: v for k, v in tr.items()
                                                                      if k != "conformal_by_alpha"})


def test_threshold_for_false_alarms_hits_the_requested_cost():
    _, _, test, _ = _world(True)
    for budget in (0.1, 0.5, 1.5):
        t = threshold_for_false_alarms(test, budget)
        got = operating_point(test, t, with_oce=False)["fp_per_image"]
        assert got <= budget + 1e-9 and got >= budget - 2.0 / test.n_images
    assert threshold_for_false_alarms(test, 0.0) == 1.0
    assert threshold_for_false_alarms(test, 1e9) == 0.0


def test_a_ranking_flip_caused_only_by_operating_point_disappears_at_matched_cost():
    """Same detector, two thresholds: a ranking 'flip' between them is purely
    an operating-point effect and must vanish once both are moved to the same
    false-alarm budget.
    """
    _, _, test, bw = _world(True)
    strict = {"point": operating_point(test, 0.80), "reps": metric_replicates(test, 0.80, 10, bw)}
    loose = {"point": operating_point(test, 0.10), "reps": metric_replicates(test, 0.10, 10, bw)}
    own = ranking_comparison({"strict": strict, "loose": loose})
    assert own["rankings"]["ma_ece"][0] == "loose"                  # the strict copy misses far more
    t = threshold_for_false_alarms(test, 0.3)
    same = {"point": operating_point(test, t), "reps": metric_replicates(test, t, 10, bw)}
    matched = ranking_comparison({"strict": same, "loose": same})
    assert not matched["any_ranking_flip_vs_dece"]
    assert all(row["dece"]["a"] == row["dece"]["b"] for row in matched["pairwise"])
