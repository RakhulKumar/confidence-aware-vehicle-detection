"""Deep-ensemble pseudo-detector built from cached raw candidates of several
training seeds of the same detector.   Used by scripts/15_build_ensemble.py.

Per image, candidates of all seeds are pooled and clustered class-agnostically:

  * candidates are visited in descending confidence; an unassigned candidate
    opens a cluster and becomes its leader;
  * the cluster then takes, from each OTHER seed, the highest-confidence
    unassigned candidate whose IoU with the leader is >= iou_threshold
    (at most one member per seed);
  * fused confidence = (sum of member confidences) / number of seeds, so a
    seed that put no box there contributes 0 -- disagreement between seeds
    lowers the score, which is the ensemble's uncertainty signal;
  * fused box = confidence-weighted mean of the member boxes;
  * fused class = class of the leader (the highest-confidence member).

The output is an ordinary RecordSet, saved in the same .npz format as
scripts/05 writes, so every later stage treats the ensemble as one detector.
"""
from __future__ import annotations

import numpy as np

from confdet.data.occlusion import iou_matrix
from confdet.evaluation.records import ImageRecord, RecordSet


def fuse_image(xyxy_per_seed: list[np.ndarray], conf_per_seed: list[np.ndarray], cls_per_seed: list[np.ndarray],
               iou_threshold: float = 0.55) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse one image's candidates from n seeds.

    Returns (xyxy, confidence, class, n_members), one row per cluster, in
    descending fused confidence.
    """
    n_seeds = len(xyxy_per_seed)
    if n_seeds == 0:
        raise ValueError("need at least one seed")
    xyxy = np.concatenate([np.asarray(b, dtype=float).reshape(-1, 4) for b in xyxy_per_seed])
    conf = np.concatenate([np.asarray(c, dtype=float).reshape(-1) for c in conf_per_seed])
    cls = np.concatenate([np.asarray(c).reshape(-1) for c in cls_per_seed]).astype(np.int64)
    seed = np.concatenate([np.full(len(np.asarray(c).reshape(-1)), s) for s, c in enumerate(conf_per_seed)])
    n = len(conf)
    if n == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32), np.zeros(0, np.int32)
    order = np.argsort(-conf, kind="stable")
    iou = iou_matrix(xyxy, xyxy)
    assigned = np.zeros(n, dtype=bool)
    out_box, out_conf, out_cls, out_n = [], [], [], []
    for i in order:
        if assigned[i]:
            continue
        assigned[i] = True
        members, used = [i], {int(seed[i])}
        if n_seeds > 1:
            near = order[(~assigned[order]) & (iou[i, order] >= iou_threshold)]   # still in confidence order
            for j in near:
                s = int(seed[j])
                if s in used:
                    continue
                members.append(j)
                used.add(s)
                assigned[j] = True
                if len(used) == n_seeds:
                    break
        m = np.asarray(members)
        c = conf[m]
        out_box.append((c[:, None] * xyxy[m]).sum(axis=0) / c.sum() if c.sum() > 0 else xyxy[i])
        out_conf.append(c.sum() / n_seeds)
        out_cls.append(cls[i])
        out_n.append(len(m))
    o = np.argsort(-np.asarray(out_conf), kind="stable")
    return (np.asarray(out_box, dtype=np.float32)[o], np.asarray(out_conf, dtype=np.float32)[o],
            np.asarray(out_cls, dtype=np.int32)[o], np.asarray(out_n, dtype=np.int32)[o])


def ensemble_records(recordsets: list[RecordSet], member_names: list[str], iou_threshold: float = 0.55
                     ) -> tuple[RecordSet, dict]:
    """Fuse aligned record sets (same images, same annotations, same order)
    into one. Returns (record set, statistics)."""
    if len(recordsets) < 2:
        raise ValueError("an ensemble needs at least two members")
    ref = recordsets[0]
    ids = [r.image_id for r in ref.records]
    for name, rs in zip(member_names[1:], recordsets[1:]):
        if [r.image_id for r in rs.records] != ids:
            raise ValueError(f"{name} was predicted on different images than {member_names[0]}; "
                             f"every member must come from the same split")
        if list(rs.meta.get("detector_names", [])) != list(ref.meta.get("detector_names", [])):
            raise ValueError(f"{name} has different class names than {member_names[0]}")
        for a, b in zip(ref.records, rs.records):
            if a.gt_xyxy.shape != b.gt_xyxy.shape or not np.allclose(a.gt_xyxy, b.gt_xyxy) \
                    or not np.array_equal(a.gt_cls, b.gt_cls):
                raise ValueError(f"annotations of image {a.image_id} differ between {member_names[0]} and {name}")
    records, n_in, n_out, members_hist = [], 0, 0, np.zeros(len(recordsets) + 1, dtype=np.int64)
    for k, r0 in enumerate(ref.records):
        parts = [rs.records[k] for rs in recordsets]
        xyxy, conf, cls, nm = fuse_image([p.cand_xyxy for p in parts], [p.cand_conf for p in parts],
                                         [p.cand_cls for p in parts], iou_threshold)
        n_in += sum(len(p.cand_conf) for p in parts)
        n_out += len(conf)
        members_hist += np.bincount(nm, minlength=len(recordsets) + 1)
        records.append(ImageRecord(r0.image_id, r0.width, r0.height, r0.gt_xyxy, r0.gt_cls, r0.gt_occ,
                                   xyxy, conf, cls))
    meta = dict(ref.meta)
    meta.update({"ensemble_of": list(member_names), "fusion": {
        "method": "greedy class-agnostic clustering, leader = highest confidence, at most one member per seed",
        "iou_threshold": iou_threshold, "confidence": "sum of member confidences / number of seeds",
        "box": "confidence-weighted mean", "class": "class of the highest-confidence member"}})
    stats = {"n_images": len(records), "n_member_candidates": int(n_in), "n_fused_candidates": int(n_out),
             "clusters_by_number_of_members": {str(i): int(v) for i, v in enumerate(members_hist) if i > 0}}
    return RecordSet(records, meta), stats
