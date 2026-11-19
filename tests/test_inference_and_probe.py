"""Three-tier decision rule, annotation, best-frame selection and the
controlled occlusion probe. No video, model or dataset required.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.data.synthetic_occlusion import (  # noqa: E402
    analyse_probe, covered_fraction_of_box, paste_occluder, select_targets,
)
from confdet.inference.extract_best_frames import select_best_frames  # noqa: E402
from confdet.inference.infer_video import FrameResult  # noqa: E402
from confdet.inference.tiers import TieredDetection, assign_tiers  # noqa: E402
from confdet.uq.calibrator import fit_calibrator  # noqa: E402
from confdet.uq.conformal import fit_thresholds  # noqa: E402


def _d(conf, tier="accepted", ctx=0.0):
    return TieredDetection(0, 0, 10, 10, conf, 0, tier, ctx, conf)


def _frames():
    return [
        FrameResult(0, 0.0, [_d(0.9), _d(0.85)]),
        FrameResult(5, 0.5, [_d(0.9)]),
        FrameResult(10, 1.0, []),
        FrameResult(15, 1.5, [_d(0.95), _d(0.92), _d(0.88)]),
        FrameResult(20, 2.0, [_d(0.95), _d(0.9), _d(0.1, "flagged", 0.6)]),
        FrameResult(25, 2.5, [_d(0.5)]),
    ]


def test_scorers_pick_the_expected_frames():
    assert 15 in [f.frame_index for f in select_best_frames(_frames(), 2, "high_confidence", 1)]
    assert select_best_frames(_frames(), 1, "flagged_showcase", 1)[0].frame_index == 20
    assert select_best_frames(_frames(), 1, "most_detections", 1)[0].frame_index == 15
    with pytest.raises(ValueError):
        select_best_frames(_frames(), 1, "nope")


def test_min_gap_and_ordering():
    best = select_best_frames(_frames(), n=3, scorer="high_confidence", min_gap_frames=10)
    idx = [f.frame_index for f in best]
    assert idx == sorted(idx) and all(b - a >= 10 for a, b in zip(idx, idx[1:]))


def test_tiers_accept_flag_reject():
    # confident bus; weak box half behind it; weak box on open road
    xyxy = np.array([[0, 50, 100, 150], [50, 20, 150, 120], [400, 400, 440, 440]], dtype=float)
    conf = np.array([0.9, 0.10, 0.10])
    cls = np.array([2, 0, 0])
    plain = assign_tiers(xyxy, conf, cls, 0.25)
    assert [d.tier for d in plain] == ["accepted"]

    # Mondrian thresholds: lenient where context is high, strict on open road
    rng = np.random.default_rng(0)
    n = 2000
    ctx = np.where(rng.random(n) < 0.3, rng.uniform(0.2, 0.8, n), 0.0)
    s = np.where(ctx > 0, rng.uniform(0.02, 0.6, n), rng.uniform(0.3, 1.0, n))
    thr = fit_thresholds(s, np.ones(n, bool), ctx, 0.1, context_edges=[0.0, 0.05, 1.0], min_group_size=50)
    tiers = assign_tiers(xyxy, conf, cls, 0.25, thresholds=thr)
    assert [d.tier for d in tiers] == ["accepted", "flagged"]
    assert tiers[1].occlusion_context == pytest.approx(0.35) and tiers[1].class_id == 0

    with pytest.raises(ValueError, match="calibrator"):
        aware = fit_thresholds(s, np.ones(n, bool), None, 0.1, score_name="occlusion_aware")
        assign_tiers(xyxy, conf, cls, 0.25, thresholds=aware)
    assert assign_tiers(np.zeros((0, 4)), np.zeros(0), np.zeros(0), 0.25, thresholds=thr) == []


def test_annotation_writes_an_image(tmp_path):
    from PIL import Image
    from confdet.visualization.annotate import annotate_image

    src = tmp_path / "a.jpg"
    Image.new("RGB", (200, 120), (90, 90, 90)).save(src)
    dets = [TieredDetection(10, 10, 80, 60, 0.9, 0, "accepted", 0.0, 0.9),
            TieredDetection(60, 20, 150, 90, 0.1, 5, "flagged", 0.4, 0.1)]      # class id beyond names: no crash
    out = annotate_image(src, dets, ["car"], ground_truth_boxes=[(10, 10, 80, 60)], out_path=tmp_path / "o.jpg")
    assert (tmp_path / "o.jpg").exists() and out.size == (200, 120)
    assert np.array(out).std() > np.array(Image.open(src)).std()


def test_pasted_occluder_covers_exactly_the_requested_share(tmp_path):
    from PIL import Image

    base = Image.new("RGB", (400, 300), (0, 0, 0))
    occluder = Image.new("RGB", (37, 53), (255, 255, 255))
    box = (100.0, 60.0, 220.0, 180.0)
    for sev in (0.0, 0.1, 0.3, 0.5, 0.9):
        img, occ_box = paste_occluder(base, box, occluder, sev)
        out = np.array(img)
        inside = out[60:180, 100:220, 0] > 127
        assert inside.mean() == pytest.approx(sev, abs=0.02)
        assert covered_fraction_of_box(base.size, box, occluder.size, sev) == pytest.approx(sev, abs=0.02)
        if sev > 0:
            assert inside[-1, :].all() and not inside[0, :].any()      # covers from the bottom up
            # the returned box is exactly the white (pasted) region
            ys, xs = np.nonzero(out[:, :, 0] > 127)
            assert occ_box == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        else:
            assert occ_box is None
    with pytest.raises(ValueError):
        paste_occluder(base, box, occluder, 1.5)


def test_select_targets_prefers_isolated_vehicles_and_foreign_occluders(tmp_path):
    from PIL import Image

    pairs = []
    for i in range(6):
        img, lbl = tmp_path / f"i{i}.jpg", tmp_path / f"i{i}.txt"
        Image.new("RGB", (400, 300), (50, 50, 50)).save(img)
        # i0: two overlapping vehicles (not isolated); others: one isolated vehicle
        lbl.write_text("0 0.3 0.5 0.3 0.3\n0 0.4 0.55 0.3 0.3" if i == 0 else "1 0.5 0.5 0.3 0.4")
        pairs.append((img, lbl))
    targets = select_targets(pairs, n_targets=10, seed=1)
    assert len(targets) == 5 and all(t.image_path.name != "i0.jpg" for t in targets)
    assert all(t.occluder_path != t.image_path for t in targets)


def test_probe_analysis_recovers_a_known_dose_response():
    rng = np.random.default_rng(0)
    sev = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    base = rng.uniform(0.6, 0.95, 120)
    scores = np.clip(base[:, None] - 0.8 * np.array(sev)[None, :] ** 1.5 + rng.normal(0, 0.05, (120, 10)), 0, 1)
    scores[:10, 0] = 0.0                                               # 10 targets not found even unoccluded
    r = analyse_probe(scores, sev, 0.25)
    assert r["n_targets_built"] == 120 and r["n_targets"] == 110
    assert r["dose_response_monotone"] and r["slope_ci"][0] > 0
    assert r["by_severity"][0]["miss_rate"] == 0.0 and r["by_severity"][-1]["miss_rate"] > 0.5
    flat = analyse_probe(np.clip(base[:, None] + rng.normal(0, 0.05, (120, 10)), 0, 1), sev, 0.25)
    assert flat["slope_ci"][0] < 0 < flat["slope_ci"][1]
    assert "note" in analyse_probe(np.zeros((8, 10)), sev, 0.25)
