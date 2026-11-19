"""Deep-ensemble fusion (scripts/15) and the seed summary (scripts/14), checked
on hand-built cases and on simulated seeds whose behaviour is known.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.evaluation.detection_metrics import detection_summary  # noqa: E402
from confdet.evaluation.ensemble import ensemble_records, fuse_image  # noqa: E402
from confdet.evaluation.labeling import build_tables  # noqa: E402
from confdet.evaluation.records import ImageRecord, RecordSet, load_records, save_records  # noqa: E402
from confdet.evaluation.seeds import headline_numbers, seed_summary  # noqa: E402
from confdet.evaluation.simulate import SimConfig, simulate_detections, simulate_records  # noqa: E402

A = [100, 100, 200, 200]


def test_fuse_image_by_hand():
    s0 = (np.array([A, [102, 100, 200, 200], [500, 500, 560, 560]], float), np.array([0.9, 0.5, 0.2]), np.array([1, 1, 0]))
    s1 = (np.array([[104, 104, 204, 204]], float), np.array([0.6]), np.array([2]))
    s2 = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))
    xyxy, conf, cls, nm = fuse_image([s0[0], s1[0], s2[0]], [s0[1], s1[1], s2[1]], [s0[2], s1[2], s2[2]], 0.55)
    # cluster 1: leader 0.9 (seed 0) + 0.6 (seed 1); seed 2 silent -> (0.9 + 0.6) / 3
    # cluster 2: seed 0's second box on the same car cannot join cluster 1 (one member per seed) -> 0.5 / 3
    # cluster 3: the lone 0.2 box -> 0.2 / 3
    assert conf.tolist() == pytest.approx([0.5, 0.5 / 3, 0.2 / 3], abs=1e-6)
    assert nm.tolist() == [2, 1, 1]
    assert cls[0] == 1                                                       # class of the highest-confidence member
    assert xyxy[0] == pytest.approx((0.9 * np.array(A) + 0.6 * np.array([104, 104, 204, 204])) / 1.5, abs=1e-4)
    # below the IoU threshold nothing merges
    far = np.array([[150, 150, 250, 250]], float)                           # IoU with A = 2500/17500 = 0.14
    _, c2, _, n2 = fuse_image([np.array([A], float), far], [np.array([0.8]), np.array([0.8])],
                              [np.array([0]), np.array([0])], 0.55)
    assert n2.tolist() == [1, 1] and c2.tolist() == pytest.approx([0.4, 0.4])
    # empty input
    e = fuse_image([s2[0], s2[0]], [s2[1], s2[1]], [s2[2], s2[2]])
    assert len(e[1]) == 0 and e[0].shape == (0, 4)


def _seeded(base: list[ImageRecord], k: int, cfg: SimConfig) -> RecordSet:
    """Same scenes and annotations, detection noise from seed k."""
    recs = []
    for i, r in enumerate(base):
        xyxy, conf, cls = simulate_detections(r.gt_xyxy, r.gt_cls, r.gt_occ, np.random.default_rng(1000 * k + i), cfg)
        recs.append(ImageRecord(r.image_id, r.width, r.height, r.gt_xyxy, r.gt_cls, r.gt_occ, xyxy, conf, cls))
    return RecordSet(recs, {"dataset": "sim", "role": "test", "detector_names": list("abcde"),
                            "dataset_names": list("abcde")})


def test_ensemble_of_identical_members_is_the_member():
    cfg = SimConfig()
    rs = _seeded(simulate_records(30, seed=3, cfg=cfg), 0, cfg)
    fused, st = ensemble_records([rs, rs, rs], ["a", "a2", "a3"])
    for a, b in zip(rs.records, fused.records):
        o = np.argsort(-a.cand_conf, kind="stable")
        assert np.allclose(np.sort(b.cand_conf)[::-1], a.cand_conf[o], atol=1e-6)
    assert st["n_fused_candidates"] * 3 == st["n_member_candidates"]
    assert set(st["clusters_by_number_of_members"]) == {"1", "2", "3"} and \
        st["clusters_by_number_of_members"]["3"] == st["n_fused_candidates"]
    assert fused.meta["ensemble_of"] == ["a", "a2", "a3"] and fused.meta["fusion"]["iou_threshold"] == 0.55


def test_ensemble_finds_more_vehicles_and_ranks_better_than_one_seed(tmp_path):
    cfg = SimConfig()
    base = simulate_records(300, seed=8, cfg=cfg)
    members = [_seeded(base, k, cfg) for k in range(3)]
    fused, _ = ensemble_records(members, ["s0", "s1", "s2"])
    # the .npz round trip is what scripts 06-08 read
    save_records(tmp_path / "e.npz", fused)
    fused = load_records(tmp_path / "e.npz")
    t_ens = build_tables(fused.records, class_agnostic=True)
    t_one = [build_tables(m.records, class_agnostic=True) for m in members]
    ens, one = detection_summary(t_ens, 0.25), [detection_summary(t, 0.25) for t in t_one]
    # a vehicle any seed proposes is proposed by the ensemble: recall at the floor rises
    assert ens["recall_at_floor"] > max(o["recall_at_floor"] for o in one) + 0.005
    # averaging three independent noise draws ranks candidates better
    assert ens["ap50"] > np.mean([o["ap50"] for o in one])
    assert fused.meta["ensemble_of"] == ["s0", "s1", "s2"]


def _old_load_records(path):
    """The pre-v0.5 loader (indexes the NpzFile per image), kept here only as a reference."""
    import json
    z = np.load(path, allow_pickle=False)
    gp, cp = z["gt_ptr"], z["cand_ptr"]
    recs = [ImageRecord(str(i_d), int(z["sizes"][i, 0]), int(z["sizes"][i, 1]),
                        z["gt_xyxy"][gp[i]:gp[i + 1]], z["gt_cls"][gp[i]:gp[i + 1]], z["gt_occ"][gp[i]:gp[i + 1]],
                        z["cand_xyxy"][cp[i]:cp[i + 1]], z["cand_conf"][cp[i]:cp[i + 1]],
                        z["cand_cls"][cp[i]:cp[i + 1]])
            for i, i_d in enumerate(z["image_ids"].tolist())]
    return RecordSet(recs, json.loads(str(z["meta"])))


def _saved(tmp_path, k=0, n=40):
    cfg = SimConfig()
    rs = _seeded(simulate_records(n, seed=5, cfg=cfg), k, cfg)
    path = tmp_path / f"m{k}.npz"
    save_records(path, rs)
    return rs, path


def test_load_records_shares_one_array_per_field(tmp_path):
    _, path = _saved(tmp_path)
    recs = load_records(path).records
    for field in ("cand_xyxy", "cand_conf", "cand_cls", "gt_xyxy", "gt_cls", "gt_occ"):
        bases = {id(getattr(r, field).base) for r in recs}
        assert len(bases) == 1 and getattr(recs[0], field).base is not None, field


def test_load_records_round_trip_is_exact(tmp_path):
    rs, path = _saved(tmp_path)
    back = load_records(path)
    assert back.meta == rs.meta and len(back.records) == len(rs.records)
    for a, b in zip(rs.records, back.records):
        assert (a.image_id, a.width, a.height) == (b.image_id, b.width, b.height)
        for field in ("gt_xyxy", "gt_cls", "gt_occ", "cand_xyxy", "cand_conf", "cand_cls"):
            assert np.allclose(np.asarray(getattr(a, field), float), np.asarray(getattr(b, field), float)), field
            assert len(getattr(a, field)) == len(getattr(b, field))
    old = _old_load_records(path)
    for a, b in zip(old.records, back.records):
        for field in ("gt_xyxy", "gt_cls", "gt_occ", "cand_xyxy", "cand_conf", "cand_cls"):
            assert np.array_equal(getattr(a, field), getattr(b, field)) and \
                getattr(a, field).dtype == getattr(b, field).dtype


def test_ensemble_output_is_unchanged_by_the_loader_fix(tmp_path):
    paths = [_saved(tmp_path, k)[1] for k in range(3)]
    new, st_new = ensemble_records([load_records(p) for p in paths], ["a", "b", "c"])
    old, st_old = ensemble_records([_old_load_records(p) for p in paths], ["a", "b", "c"])
    assert st_new == st_old
    for a, b in zip(old.records, new.records):
        for field in ("cand_xyxy", "cand_conf", "cand_cls"):
            assert np.array_equal(getattr(a, field), getattr(b, field))


def test_ensemble_refuses_misaligned_members():
    cfg = SimConfig()
    a = _seeded(simulate_records(5, seed=1, cfg=cfg), 0, cfg)
    b = _seeded(simulate_records(5, seed=2, cfg=cfg), 1, cfg)            # different scenes
    with pytest.raises(ValueError):
        ensemble_records([a, b], ["a", "b"])
    with pytest.raises(ValueError):
        ensemble_records([a], ["a"])


def _fake_results(ap, recall, odds, cov05, flags05, cov10, d_all, dece):
    rule = lambda c, fl: {"coverage": c, "flagged_per_image": fl, "all_groups_achievable": True}  # noqa: E731
    return {
        "summary": {"standard_threshold": 0.5, "headline": {"detection": {
            "recall_at_standard": recall, "fp_per_image_at_standard": 1.4, "recall_at_floor": 0.98}}},
        "detection": {"class_agnostic": {"ap50": ap}},
        "rq1": {"logistic_miss_model": {"odds_ratio_per_0.1_occlusion": odds}, "partial_spearman_given_size_and_class": 0.2},
        "conformal": {"validity_curve": [
            {"alpha": 0.05, "global_confidence": rule(cov05, flags05)},
            {"alpha": 0.1, "global_confidence": rule(cov10, 1.0)}]},
        "rq2": {"policies": {"occlusion_aware": {"mean_recovery_delta_vs_confidence": d_all,
                                                 "mean_recovery_occluded_delta_vs_confidence": 0.01}}},
        "rq3": {"at_standard_threshold": {"dece": dece, "ma_ece": 0.16, "ma_brier": 0.2, "oce_max": 0.16,
                                          "miss_mass": 0.12}},
    }


def test_seed_summary_mean_and_sd():
    per = {"m0": headline_numbers(_fake_results(0.91, 0.86, 1.19, 0.95, 4.5, 0.90, 0.04, 0.05)),
           "m1": headline_numbers(_fake_results(0.93, 0.88, 1.21, 0.94, 4.7, 0.91, 0.03, 0.06)),
           "m2": headline_numbers(_fake_results(0.92, 0.87, 1.20, 0.96, 4.3, 0.89, 0.05, 0.04))}
    s = seed_summary(per)
    m = s["metrics"]
    assert s["n_members"] == 3
    assert m["ap50_class_agnostic"]["mean"] == pytest.approx(0.92)
    assert m["ap50_class_agnostic"]["sd"] == pytest.approx(0.01)
    assert m["conformal_a0.05.global_confidence.coverage"]["mean"] == pytest.approx(0.95)
    assert m["conformal_a0.05.global_confidence.flags_per_image"]["sd"] == pytest.approx(0.2)
    assert m["conformal_a0.1.global_confidence.coverage"]["min"] == pytest.approx(0.89)
    assert m["rq2.occlusion_aware.delta_all"]["max"] == pytest.approx(0.05)
    assert m["rq1_odds_ratio_per_0.1_occlusion"]["values"] == {"m0": 1.19, "m1": 1.21, "m2": 1.20}
    # a quantity a member lacks is skipped, and n says so
    assert m["rq2.mondrian_rank.delta_all"]["n"] == 0 and m["rq2.mondrian_rank.delta_all"]["mean"] is None
    assert seed_summary({"only": per["m0"]})["metrics"]["dece"]["sd"] is None
