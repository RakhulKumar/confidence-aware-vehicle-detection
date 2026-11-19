"""NMS pilot (v0.6 Phase 1b item 6): suppression variants applied to pre-NMS candidates,
and the mechanism behind each missed vehicle.   Used by scripts/18_nms_pilot.py.

Suppression is per class, as Ultralytics' default NMS (agnostic_nms=False) that produced
our cached predictions:
  hard_nms      greedy: keep the highest box, drop same-class boxes with IoU > iou_thr.
  soft_nms      Gaussian soft-NMS (Bodla et al. 2017): instead of dropping, multiply each
                remaining same-class score by exp(-IoU^2 / sigma); boxes below score_thr go.
Both return indices (and, for soft-NMS, the decayed scores), highest score first, at most max_det.

miss_mechanism  for every vehicle missed by the standard output at threshold tau (class-
                agnostic), the first that applies:
                  removed_by_nms           a pre-NMS box with IoU >= iou_thr and confidence >= tau existed
                  kept_below_threshold     the standard output proposed it, below tau
                  suppressed_low_confidence  only pre-NMS boxes below tau overlapped it
                  never_proposed           no pre-NMS box (within the per-image cap) overlapped it
"""
from __future__ import annotations

import numpy as np

from confdet.data.occlusion import bucket_indices, iou_matrix
from confdet.evaluation.records import ImageRecord, RecordSet

MECHANISMS = ("removed_by_nms", "kept_below_threshold", "suppressed_low_confidence", "never_proposed")


def hard_nms(xyxy, conf, cls, iou_thr: float, max_det: int = 300) -> np.ndarray:
    xyxy, conf, cls = np.asarray(xyxy, float).reshape(-1, 4), np.asarray(conf, float), np.asarray(cls)
    keep = []
    for c in np.unique(cls):
        idx = np.nonzero(cls == c)[0]
        idx = idx[np.argsort(-conf[idx], kind="stable")]
        iou = iou_matrix(xyxy[idx], xyxy[idx]) if len(idx) else np.zeros((0, 0))
        alive = np.ones(len(idx), bool)
        for i in range(len(idx)):
            if not alive[i]:
                continue
            keep.append(idx[i])
            alive[i + 1:] &= ~(iou[i, i + 1:] > iou_thr)
    keep = np.asarray(keep, int)
    return keep[np.argsort(-conf[keep], kind="stable")][:max_det] if len(keep) else keep


def soft_nms(xyxy, conf, cls, sigma: float = 0.5, score_thr: float = 0.001, max_det: int = 300
             ) -> tuple[np.ndarray, np.ndarray]:
    xyxy, conf, cls = np.asarray(xyxy, float).reshape(-1, 4), np.asarray(conf, float), np.asarray(cls)
    out_idx, out_score = [], []
    for c in np.unique(cls):
        idx = np.nonzero(cls == c)[0]
        s = conf[idx].copy()
        iou = iou_matrix(xyxy[idx], xyxy[idx]) if len(idx) else np.zeros((0, 0))
        alive = np.ones(len(idx), bool)
        while alive.any():
            i = int(np.argmax(np.where(alive, s, -np.inf)))
            if s[i] < score_thr:
                break
            out_idx.append(idx[i]); out_score.append(s[i])
            alive[i] = False
            s = np.where(alive, s * np.exp(-(iou[i] ** 2) / sigma), s)
            alive &= s >= score_thr
    out_idx, out_score = np.asarray(out_idx, int), np.asarray(out_score, float)
    o = np.argsort(-out_score, kind="stable")[:max_det]
    return out_idx[o], out_score[o]


def apply_suppression(rs: RecordSet, method: str, **kw) -> RecordSet:
    """New record set with every image's candidates passed through `method` (hard / soft)."""
    recs = []
    for r in rs.records:
        if method == "hard":
            k = hard_nms(r.cand_xyxy, r.cand_conf, r.cand_cls, **kw)
            conf = r.cand_conf[k]
        elif method == "soft":
            k, conf = soft_nms(r.cand_xyxy, r.cand_conf, r.cand_cls, **kw)
        else:
            raise ValueError(method)
        recs.append(ImageRecord(r.image_id, r.width, r.height, r.gt_xyxy, r.gt_cls, r.gt_occ,
                                r.cand_xyxy[k], np.asarray(conf, np.float32), r.cand_cls[k]))
    return RecordSet(recs, {**rs.meta, "suppression": {"method": method, **kw}})


def miss_mechanism(pre: RecordSet, std_tables, tau: float, bucket_edges, iou_thr: float = 0.5) -> dict:
    """std_tables: class-agnostic tables of the STANDARD (post-NMS) output of the same images."""
    found = std_tables.g_proposed & (std_tables.g_score >= tau)
    mech = np.full(std_tables.n_gt, "", dtype=object)
    off = 0
    for r in pre.records:
        ng = len(r.gt_cls)
        rows = np.arange(off, off + ng)
        if ng:
            iou = iou_matrix(np.asarray(r.gt_xyxy, float), np.asarray(r.cand_xyxy, float)) if len(r.cand_conf) \
                else np.zeros((ng, 0))
            hit = iou >= iou_thr
            best = np.where(hit, np.asarray(r.cand_conf, float)[None, :], -1.0).max(axis=1) if iou.shape[1] \
                else np.full(ng, -1.0)
            for j, g in enumerate(rows):
                if found[g]:
                    continue
                if best[j] >= tau:
                    mech[g] = "removed_by_nms"
                elif std_tables.g_proposed[g]:
                    mech[g] = "kept_below_threshold"
                elif best[j] >= 0:
                    mech[g] = "suppressed_low_confidence"
                else:
                    mech[g] = "never_proposed"
        off += ng
    if off != std_tables.n_gt:
        raise ValueError("pre-NMS records and the standard tables do not hold the same vehicles")
    missed = ~found
    b = bucket_indices(std_tables.g_occ, bucket_edges)
    by_bucket = []
    for k in range(len(bucket_edges) - 1):
        m = missed & (b == k)
        n = int(m.sum())
        by_bucket.append({"lo": float(bucket_edges[k]), "hi": float(bucket_edges[k + 1]),
                          "n_objects": int((b == k).sum()), "n_missed": n,
                          "shares": {x: float((mech[m] == x).mean()) if n else None for x in MECHANISMS}})
    n = int(missed.sum())
    return {"tau": tau, "n_missed": n, "counts": {x: int((mech[missed] == x).sum()) for x in MECHANISMS},
            "shares": {x: float((mech[missed] == x).mean()) if n else None for x in MECHANISMS},
            "by_annotated_occlusion": by_bucket}
