"""Synthetic-data tests for occlusion severity — no dataset required.
Run with: pytest tests/test_occlusion.py -v   (or: python tests/run_tests.py occlusion)
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.data.occlusion import (  # noqa: E402
    Box,
    bucket_index,
    compute_occlusion_for_image,
    compute_occlusion_severity,
)


def test_no_occluders_gives_zero_severity():
    target = Box(0, 0, 10, 10)
    assert compute_occlusion_severity(target, []) == 0.0


def test_fully_covered_target_gives_severity_one():
    # Occluder sits lower in frame (larger y2) => "nearer" under bottom_y proxy,
    # and fully contains the target.
    target = Box(0, 0, 10, 10)
    occluder = Box(-5, -5, 15, 20)
    assert compute_occlusion_severity(target, [occluder]) == pytest.approx(1.0)


def test_half_covered_target_gives_half_severity():
    target = Box(0, 0, 10, 10)
    # Occluder covers the right half of target and is lower (nearer).
    occluder = Box(5, -5, 20, 20)
    assert compute_occlusion_severity(target, [occluder]) == pytest.approx(0.5)


def test_farther_box_never_counts_as_occluder():
    target = Box(0, 0, 10, 10)
    # This box overlaps target fully in space but is *higher* in the frame
    # (smaller y2), so under the bottom_y proxy it is farther away and must
    # not be treated as occluding the target.
    farther = Box(-5, -5, 15, 5)
    assert compute_occlusion_severity(target, [farther]) == pytest.approx(0.0)


def test_area_proxy_behaves_independently_of_position():
    target = Box(0, 0, 10, 10)  # area 100
    bigger = Box(5, 5, 25, 25)  # area 400, overlaps bottom-right quadrant
    sev_area = compute_occlusion_severity(target, [bigger], proxy="area")
    assert sev_area > 0.0


def test_severity_is_bounded_in_unit_interval():
    target = Box(0, 0, 10, 10)
    # Multiple overlapping occluders shouldn't push severity above 1.0 via
    # double counting — that's exactly what the union-area sweep guards against.
    occluders = [Box(2, -5, 20, 20), Box(-5, -5, 8, 20), Box(0, -5, 15, 20)]
    sev = compute_occlusion_severity(target, occluders)
    assert 0.0 <= sev <= 1.0
    assert sev == pytest.approx(1.0)  # union of these three fully covers target


def test_zero_area_box_returns_zero_not_nan():
    degenerate = Box(5, 5, 5, 5)
    assert compute_occlusion_severity(degenerate, [Box(0, 0, 10, 10)]) == 0.0


def test_compute_for_image_excludes_self():
    boxes = [Box(0, 0, 10, 10), Box(0, 0, 10, 20)]
    severities = compute_occlusion_for_image(boxes)
    assert len(severities) == 2
    # box 1 (taller, lower bottom_y) occludes box 0 fully; box 1 has nothing
    # nearer than itself, so its own severity is 0.
    assert severities[0] == pytest.approx(1.0)
    assert severities[1] == pytest.approx(0.0)


def test_bucket_index_edges():
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cases = [(0.0, 0), (0.19, 0), (0.2, 1), (0.55, 2), (0.999, 4), (1.0, 4)]  # top edge is closed
    for severity, expected_bucket in cases:
        assert bucket_index(severity, edges) == expected_bucket


# ---------------------------------------------------------------------------
# Vectorised implementation must agree exactly with the reference one.
# ---------------------------------------------------------------------------
def _random_boxes(rng, n, size=100.0):
    x1 = rng.uniform(0, size, n)
    y1 = rng.uniform(0, size, n)
    w = rng.uniform(2, size / 2, n)
    h = rng.uniform(2, size / 2, n)
    return [Box(float(a), float(b), float(a + c), float(b + d)) for a, b, c, d in zip(x1, y1, w, h)]


def test_vectorised_severity_matches_reference_on_random_scenes():
    import numpy as np
    from confdet.data.occlusion import boxes_to_array, occlusion_severity_array

    rng = np.random.default_rng(0)
    for proxy in ("bottom_y", "area"):
        for n in (0, 1, 2, 5, 12, 25):
            boxes = _random_boxes(rng, n)
            ref = compute_occlusion_for_image(boxes, proxy=proxy)
            vec = occlusion_severity_array(boxes_to_array(boxes), proxy=proxy)
            assert len(vec) == n
            for a, b in zip(ref, vec):
                assert a == pytest.approx(b, abs=1e-9)


def test_occlusion_context_uses_only_confident_nearer_non_duplicate_boxes():
    import numpy as np
    from confdet.data.occlusion import occlusion_context

    # 0: confident bus low in frame (near).  1: weak box half hidden behind it.
    # 2: weak duplicate of the bus (IoU ~0.9). 3: weak box far away from all.
    xyxy = np.array([
        [0, 50, 100, 150],
        [50, 20, 150, 120],
        [2, 52, 100, 151],
        [300, 300, 340, 340],
    ], dtype=float)
    conf = np.array([0.9, 0.08, 0.05, 0.04])
    o_hat, dup = occlusion_context(xyxy, conf, context_conf=0.25, duplicate_iou=0.5)

    # box 1: overlap with box 0 is x 50..100, y 50..120 = 3500 of its 10000 px
    assert o_hat[1] == pytest.approx(0.35)
    assert 0.0 < dup[1] < 0.5
    # box 2 is a duplicate: large dup, but NOT counted as occluded
    assert dup[2] > 0.9
    assert o_hat[2] == 0.0
    # box 3 and the confident box itself have no context
    assert o_hat[3] == 0.0 and dup[3] == 0.0
    assert o_hat[0] == 0.0 and dup[0] == 0.0


def test_occlusion_context_ignores_low_confidence_occluders():
    import numpy as np
    from confdet.data.occlusion import occlusion_context

    xyxy = np.array([[0, 50, 100, 150], [50, 20, 150, 120]], dtype=float)
    conf = np.array([0.10, 0.08])            # nothing clears the context threshold
    o_hat, dup = occlusion_context(xyxy, conf, context_conf=0.25)
    assert (o_hat == 0).all() and (dup == 0).all()


def test_bucket_indices_matches_scalar_version():
    import numpy as np
    from confdet.data.occlusion import bucket_indices

    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    vals = np.array([0.0, 0.1999, 0.2, 0.55, 0.8, 1.0])
    assert bucket_indices(vals, edges).tolist() == [bucket_index(v, edges) for v in vals]


def test_context_at_a_box_ignores_a_detection_of_that_same_vehicle():
    """An unproposed (e.g. misnamed) vehicle must not be filed under "heavily
    occluded" just because the confident box ON it ends a pixel lower.
    """
    import numpy as np
    from confdet.data.occlusion import occlusion_context_at_boxes

    gt = np.array([[100, 100, 200, 200]], dtype=float)
    same_vehicle = np.array([[101, 101, 201, 202]], dtype=float)        # IoU ~0.96, bottom edge lower
    assert occlusion_context_at_boxes(gt, same_vehicle, np.array([0.9]), 0.25)[0] == 0.0
    # a genuinely different, nearer vehicle covering its lower half still counts
    occluder = np.array([[60, 150, 240, 300]], dtype=float)
    assert occlusion_context_at_boxes(gt, occluder, np.array([0.9]), 0.25)[0] == pytest.approx(0.5)
    both = np.vstack([same_vehicle, occluder])
    assert occlusion_context_at_boxes(gt, both, np.array([0.9, 0.9]), 0.25)[0] == pytest.approx(0.5)
