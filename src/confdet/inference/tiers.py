"""The deployed decision rule: every raw candidate ends in one of three tiers.

  accepted   confidence >= standard threshold. The detector's normal output,
             never altered by the post-hoc layer.
  flagged    below the standard threshold but kept by the conformal rule:
             "possible vehicle, likely occluded -- look here". Accepted +
             flagged together carry the calibrated recall guarantee.
  rejected   everything else (not returned).

Uses only detector outputs (confidence, boxes) -- no labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from confdet.data.occlusion import occlusion_context
from confdet.uq.conformal import keep_mask


@dataclass
class TieredDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    tier: str                 # "accepted" | "flagged"
    occlusion_context: float  # o_hat: share of the box behind confident nearer detections
    score: float              # the score the conformal rule thresholded


def assign_tiers(xyxy, conf, cls, std_threshold: float, thresholds=None, calibrator=None,
                 context_conf: float | None = None, duplicate_iou: float = 0.5,
                 proxy: str = "bottom_y") -> list[TieredDetection]:
    """thresholds: a ConformalThresholds (or None for plain thresholding).
    If its score is "occlusion_aware", `calibrator` must be given.
    """
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    conf = np.asarray(conf, dtype=float).reshape(-1)
    cls = np.asarray(cls, dtype=int).reshape(-1)
    ctx, dup = occlusion_context(xyxy, conf, std_threshold if context_conf is None else context_conf,
                                 duplicate_iou, proxy)
    accepted = conf >= std_threshold
    score = conf
    kept = np.zeros(len(conf), dtype=bool)
    if thresholds is not None:
        if thresholds.score_name == "occlusion_aware":
            if calibrator is None:
                raise ValueError("these thresholds apply to the occlusion-aware score; pass the calibrator")
            score = calibrator.predict(conf, ctx, dup)
        kept = keep_mask(score, ctx if len(thresholds.context_edges) > 2 else None, thresholds)
    out = []
    for k in np.argsort(-conf, kind="stable"):
        if accepted[k] or kept[k]:
            out.append(TieredDetection(*[float(v) for v in xyxy[k]], float(conf[k]), int(cls[k]),
                                       "accepted" if accepted[k] else "flagged", float(ctx[k]), float(score[k])))
    return out
