"""NMS pilot (evaluation/nms.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.evaluation.labeling import build_tables  # noqa: E402
from confdet.evaluation.nms import apply_suppression, hard_nms, miss_mechanism, soft_nms  # noqa: E402
from confdet.evaluation.records import ImageRecord, RecordSet  # noqa: E402

B = np.array([[0, 0, 100, 100], [5, 0, 105, 100], [60, 0, 160, 100], [300, 300, 400, 400]], float)
C = np.array([0.9, 0.8, 0.7, 0.6])


def test_hard_nms_is_per_class_and_iou_thresholded():
    # IoU(0,1) = 0.905, IoU(0,2) = 0.25, IoU(1,2) = 0.29
    assert sorted(hard_nms(B, C, np.zeros(4, int), 0.7).tolist()) == [0, 2, 3]
    assert sorted(hard_nms(B, C, np.zeros(4, int), 0.2).tolist()) == [0, 3]
    assert sorted(hard_nms(B, C, np.array([0, 1, 0, 0]), 0.7).tolist()) == [0, 1, 2, 3]   # other class survives
    assert sorted(hard_nms(B, C, np.zeros(4, int), 1.0).tolist()) == [0, 1, 2, 3]          # IoU 1.0 = no suppression
    assert hard_nms(B, C, np.zeros(4, int), 0.7, max_det=2).tolist() == [0, 2]


def test_soft_nms_decays_instead_of_dropping():
    idx, s = soft_nms(B, C, np.zeros(4, int), sigma=0.5)
    d = dict(zip(idx.tolist(), s.tolist()))
    assert d[0] == pytest.approx(0.9) and d[3] == pytest.approx(0.6)
    iou01, iou02, iou12 = 9500 / 10500, 4000 / 16000, 4500 / 15500
    assert d[2] == pytest.approx(0.7 * np.exp(-iou02 ** 2 / 0.5), rel=1e-6)            # decayed once, by box 0
    # box 1 is decayed by box 0 and, after box 2 is picked, by box 2 as well
    assert d[1] == pytest.approx(0.8 * np.exp(-iou01 ** 2 / 0.5) * np.exp(-iou12 ** 2 / 0.5), rel=1e-6)
    assert d[1] < d[2] < 0.7                                                            # box 2 decayed less
    assert list(s) == sorted(s, reverse=True)
    idx2, _ = soft_nms(B, C, np.zeros(4, int), sigma=0.5, score_thr=0.5)
    assert 1 not in idx2.tolist()                                                       # fell below the floor


def _rec(gt, cands, name="a.jpg"):
    g = np.array(gt, float).reshape(-1, 4)
    c = np.array(cands, float).reshape(-1, 5)
    return ImageRecord(name, 1000, 1000, g, np.zeros(len(g), int), np.zeros(len(g)), c[:, :4], c[:, 4],
                       np.zeros(len(c), int))


def test_miss_mechanism_assigns_each_case():
    gt = [(0, 0, 100, 100), (300, 0, 400, 100), (600, 0, 700, 100), (800, 0, 900, 100), (0, 500, 100, 600)]
    pre = RecordSet([_rec(gt, [(0, 0, 100, 100, 0.9),        # found
                               (300, 0, 400, 100, 0.8),      # confident pre-NMS box ...
                               (600, 0, 700, 100, 0.3),      # kept, below tau
                               (800, 0, 900, 100, 0.2)])], {})  # pre-NMS only, below tau; last vehicle: nothing
    std = RecordSet([_rec(gt, [(0, 0, 100, 100, 0.9), (600, 0, 700, 100, 0.3)])], {})  # ... removed by NMS
    t = build_tables(std.records, class_agnostic=True)
    m = miss_mechanism(pre, t, 0.5, [0, 0.5, 1.0])
    assert m["n_missed"] == 4
    assert m["counts"] == {"removed_by_nms": 1, "kept_below_threshold": 1, "suppressed_low_confidence": 1,
                           "never_proposed": 1}
    assert m["by_annotated_occlusion"][0]["n_missed"] == 4


def test_apply_suppression_round_trip():
    rs = RecordSet([_rec([(0, 0, 100, 100)], [tuple(b) + (c,) for b, c in zip(B, C)])], {"dataset": "d"})
    h = apply_suppression(rs, "hard", iou_thr=0.7)
    assert len(h.records[0].cand_conf) == 3 and h.meta["suppression"]["method"] == "hard"
    s = apply_suppression(rs, "soft", sigma=0.5)
    assert len(s.records[0].cand_conf) == 4 and s.records[0].cand_conf[0] == pytest.approx(0.9)
