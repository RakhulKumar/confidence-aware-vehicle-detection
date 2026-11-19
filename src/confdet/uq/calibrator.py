"""Occlusion-aware candidate score.  docs/theoretical_foundations.md section 5.

A confidence threshold ranks candidates by confidence alone. But the chance
that a candidate is a real vehicle can depend on where it sits: a weak box
half hidden behind a confidently detected bus is a different thing from an
equally weak box on open road, and both differ from a weak second box on a
vehicle that is already detected. This module fits

    P(true positive | confidence, occlusion context)
        = sigmoid( b0 + b1*z + b2*o_hat + b3*dup + b4*z*o_hat + b5*z*dup ),
    z = logit(confidence)

on the calibration role, by logistic regression, using only quantities the
deployed system has (data/occlusion.occlusion_context).

Two properties make this a fair comparison rather than a tuned one:

  * NESTING. With b2 = b3 = b4 = b5 = 0 the score is a monotone function of
    confidence, i.e. exactly the baseline ranking. The baseline is a special
    case of this model, so with enough calibration data the fitted score
    cannot rank worse than confidence except by estimation noise.
  * NO FREE LUNCH FOR MONOTONE RECALIBRATION. Temperature scaling, Platt
    scaling and isotonic regression all leave the ranking of candidates
    unchanged, so their recall-vs-false-alarm curve is identical to raw
    confidence. Only side information can move that curve. `features="confidence"`
    fits the Platt baseline so this can be checked on real data.

Whether occlusion context actually carries information for a given detector
is an empirical question; RQ2 measures it and reports it either way.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from confdet.evaluation.stats import logistic_irls, sigmoid

FEATURE_SETS = {
    "confidence": ["z"],
    "occlusion_aware": ["z", "ctx", "dup", "z*ctx", "z*dup"],
    # ablation sets (scripts/13_calibrator_ablation.py); the deployed default is posthoc.score_features
    "z_dup": ["z", "dup"],
    "z_ctx": ["z", "ctx", "z*ctx"],
    # dupS = dup if dup >= duplicate_iou else 0. Occlusion context only counts confident boxes with
    # IoU < duplicate_iou, so dupS and ctx are built from disjoint sets of neighbouring boxes.
    "full_strict": ["z", "ctx", "dupS", "z*ctx", "z*dupS"],
}


def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def strict_duplicate(dup, duplicate_iou: float) -> np.ndarray:
    """dupS: the duplicate feature restricted to true duplicates (IoU >= duplicate_iou)."""
    dup = np.asarray(dup, dtype=float)
    return np.where(dup >= duplicate_iou, dup, 0.0)


def _design(conf, ctx, dup, features: list[str], duplicate_iou: float = 0.5) -> np.ndarray:
    z = _logit(conf)
    dup = np.asarray(dup, dtype=float)
    base = {"z": z, "ctx": np.asarray(ctx, dtype=float), "dup": dup,
            "dupS": strict_duplicate(dup, duplicate_iou)}
    cols = []
    for f in features:
        if "*" in f:
            a, b = f.split("*")
            cols.append(base[a] * base[b])
        else:
            cols.append(base[f])
    return np.column_stack(cols) if cols else np.zeros((len(z), 0))


@dataclass
class CandidateCalibrator:
    features: list[str]
    mean: list[float]
    std: list[float]
    coef: list[float]             # intercept first, then one per (standardised) feature
    n_fit: int = 0
    n_positive: int = 0
    name: str = "occlusion_aware"
    duplicate_iou: float = 0.5     # only used by the dupS feature

    def predict(self, conf, ctx=None, dup=None) -> np.ndarray:
        conf = np.asarray(conf, dtype=float)
        zeros = np.zeros(len(conf))
        X = _design(conf, zeros if ctx is None else ctx, zeros if dup is None else dup, self.features,
                    self.duplicate_iou)
        Xs = (X - np.asarray(self.mean)) / np.asarray(self.std)
        return sigmoid(self.coef[0] + Xs @ np.asarray(self.coef[1:]))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CandidateCalibrator":
        return CandidateCalibrator(**d)


def fit_calibrator(conf, ctx, dup, is_tp, features: str = "occlusion_aware",
                   weights=None, l2: float = 1e-4, duplicate_iou: float = 0.5) -> CandidateCalibrator:
    """Fit on calibration CANDIDATES (one row per raw candidate, labelled
    true/false positive by labeling.build_tables).
    duplicate_iou: threshold of the dupS feature (posthoc.duplicate_iou).
    """
    if features not in FEATURE_SETS:
        raise ValueError(f"features must be one of {sorted(FEATURE_SETS)}, got {features!r}")
    names = FEATURE_SETS[features]
    y = np.asarray(is_tp, dtype=float)
    if len(y) < 50 or y.sum() < 10 or (len(y) - y.sum()) < 10:
        raise ValueError(
            f"too little calibration data to fit a candidate score "
            f"({len(y)} candidates, {int(y.sum())} true positives, {int(len(y) - y.sum())} false positives)")
    X = _design(conf, ctx, dup, names, duplicate_iou)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Xs = np.column_stack([np.ones(len(y)), (X - mean) / std])
    beta = logistic_irls(Xs, y, weights=weights, l2=l2)
    return CandidateCalibrator(features=names, mean=mean.tolist(), std=std.tolist(),
                               coef=beta.tolist(), n_fit=int(len(y)), n_positive=int(y.sum()),
                               name=features, duplicate_iou=float(duplicate_iou))
