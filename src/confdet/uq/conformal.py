"""Miss-aware split conformal prediction for a frozen detector.
docs/theoretical_foundations.md section 4.

What is being guaranteed
------------------------
The object of study is the missed vehicle, so the conformal statement has to
be about ground-truth OBJECTS, including the ones the detector produced
nothing for. For every calibration object j let

    S_j = score of the candidate matched to it     (labeling.build_tables)
    S_j = -infinity                                 if nothing was proposed

and let the system keep every candidate whose score is >= t. Object j is
then found iff S_j >= t. With n calibration objects and
k = floor(alpha * (n + 1)), choose

    t_hat = k-th smallest value of {S_1, ..., S_n}.

If calibration and test objects are exchangeable, a new object is found with
probability >= 1 - alpha:

    P(S_test >= t_hat) >= 1 - alpha.

Two things can make the target unreachable, and both are reported instead of
hidden:

  * k = 0 (too few calibration objects to certify this alpha);
  * the k-th smallest value is -infinity, i.e. more than a fraction alpha of
    the objects have no candidate at all. No threshold on an existing
    candidate can recover an object that was never proposed. The smallest
    certifiable miss rate is then (n_unproposed + 1) / (n + 1).

The previous implementation of this file calibrated only on objects the
detector had already found and never used the result to make a decision; it
could say nothing about misses. This is the replacement.

Occlusion conditioning (Mondrian)
---------------------------------
A single threshold only controls the miss rate on average. Misses concentrate
on occluded vehicles, so the average can be met while occluded vehicles are
missed far more often than alpha. Mondrian (group-conditional) conformal
prediction fits one threshold per stratum and gives the guarantee within each
stratum. The strata here are bins of the occlusion CONTEXT o_hat, which is
computed from the detector's own confident detections and so exists at test
time (data/occlusion.py). Strata are never defined from ground-truth
occlusion: a rule that needs the labels cannot be deployed.

A stratum with too few calibration objects is merged with its neighbour
before thresholds are fitted. Merging depends only on stratum sizes, not on
scores, so each merged group keeps a valid guarantee.

Exchangeability caveat: objects of one image are not independent. The
guarantee is exact for exchangeable objects and is checked empirically on the
test role with image-level bootstrap intervals (evaluation/rq2_recovery.py).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

NEG_INF = float("-inf")


@dataclass
class GroupThreshold:
    threshold: float            # keep candidates with score >= threshold (-inf: keep all)
    n: int                      # calibration objects in the group
    n_unproposed: int
    achievable: bool            # can the target alpha be certified in this group?
    min_alpha: float            # smallest certifiable miss rate, (n_unproposed + 1) / (n + 1)
    reason: str = ""            # why not achievable, if not


@dataclass
class ConformalThresholds:
    alpha: float
    score_name: str                       # "confidence" | "occlusion_aware"
    context_edges: list[float]            # base strata over o_hat; [0.0, 1.0] = one global stratum
    group_of_stratum: list[int]           # merged-group id of each base stratum
    groups: list[GroupThreshold]
    # sorted calibration object scores per group (-inf for unproposed objects);
    # this is all the rank transform needs
    calib_scores: list[list[float]] = field(default_factory=list)

    @property
    def is_conditional(self) -> bool:
        return len(self.groups) > 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["calib_scores"] = [[None if s == NEG_INF else float(s) for s in g] for g in self.calib_scores]
        for g in d["groups"]:
            if g["threshold"] == NEG_INF:
                g["threshold"] = None
        return d

    @staticmethod
    def from_dict(d: dict) -> "ConformalThresholds":
        groups = [GroupThreshold(**{**g, "threshold": NEG_INF if g["threshold"] is None else g["threshold"]})
                  for g in d["groups"]]
        scores = [[NEG_INF if s is None else float(s) for s in g] for g in d.get("calib_scores", [])]
        return ConformalThresholds(alpha=d["alpha"], score_name=d["score_name"],
                                   context_edges=list(d["context_edges"]),
                                   group_of_stratum=list(d["group_of_stratum"]),
                                   groups=groups, calib_scores=scores)


def _effective(scores: np.ndarray, proposed: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(proposed, dtype=bool), np.asarray(scores, dtype=float), NEG_INF)


def miss_aware_quantile(scores: np.ndarray, proposed: np.ndarray, alpha: float) -> GroupThreshold:
    """Threshold t_hat with P(S_test >= t_hat) >= 1 - alpha (see module docstring)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    eff = np.sort(_effective(scores, proposed))
    n = len(eff)
    n_unprop = int(np.sum(eff == NEG_INF))
    min_alpha = (n_unprop + 1) / (n + 1)
    k = int(np.floor(alpha * (n + 1) + 1e-12))
    if k < 1:
        return GroupThreshold(NEG_INF, n, n_unprop, False, min_alpha,
                              f"only {n} calibration objects; alpha={alpha} needs at least "
                              f"{int(np.ceil(1 / alpha)) - 1}")
    t = float(eff[k - 1])
    if t == NEG_INF:
        return GroupThreshold(NEG_INF, n, n_unprop, False, min_alpha,
                              f"{n_unprop}/{n} objects have no candidate at all; the smallest "
                              f"certifiable miss rate is {min_alpha:.3f}")
    return GroupThreshold(t, n, n_unprop, True, min_alpha)


def certifiability_by_group(thr: "ConformalThresholds") -> list[dict]:
    """Why a (Mondrian) rule is or is not certifiable, one row per merged group:
    o_hat range of the group, calib objects n, objects never proposed, the group's
    smallest certifiable miss rate (n_unproposed + 1) / (n + 1) against the target alpha.
    Reads the fitted thresholds only; changes nothing.
    """
    e = thr.context_edges
    rows = []
    for g, gt in enumerate(thr.groups):
        strata = [s for s, m in enumerate(thr.group_of_stratum) if m == g]
        rows.append({"group": g, "o_hat_lo": float(e[min(strata)]), "o_hat_hi": float(e[max(strata) + 1]),
                     "n_calib": int(gt.n), "n_never_proposed": int(gt.n_unproposed),
                     "share_never_proposed": float(gt.n_unproposed / gt.n) if gt.n else None,
                     "min_certifiable_alpha": float(gt.min_alpha), "target_alpha": float(thr.alpha),
                     "certifiable": bool(gt.achievable), "reason": gt.reason})
    return rows


def stratum_indices(context: np.ndarray, edges: list[float]) -> np.ndarray:
    e = np.asarray(edges, dtype=float)
    idx = np.searchsorted(e, np.asarray(context, dtype=float), side="right") - 1
    return np.clip(idx, 0, len(e) - 2).astype(int)


def merge_small_strata(counts: list[int], min_size: int) -> list[int]:
    """Map each base stratum to a merged-group id so every group has at least
    `min_size` members (if the total allows it). Only ADJACENT strata are
    merged: sparse high-occlusion strata fold into the next lower one, and a
    sparse lowest stratum folds upward.
    """
    n = len(counts)
    blocks = [[i] for i in range(n)]
    sizes = list(counts)
    i = len(blocks) - 1
    while i > 0:                                  # top-down: fold into the lower neighbour
        if sizes[i] < min_size:
            blocks[i - 1] += blocks[i]
            sizes[i - 1] += sizes[i]
            del blocks[i], sizes[i]
        i -= 1
    while len(blocks) > 1 and sizes[0] < min_size:  # lowest block still small: fold upward
        blocks[1] = blocks[0] + blocks[1]
        sizes[1] += sizes[0]
        del blocks[0], sizes[0]
    mapping = [0] * n
    for g, block in enumerate(blocks):
        for s in block:
            mapping[s] = g
    return mapping


def fit_thresholds(
    scores: np.ndarray,
    proposed: np.ndarray,
    context: np.ndarray | None,
    alpha: float,
    context_edges: list[float] | None = None,
    min_group_size: int = 100,
    score_name: str = "confidence",
) -> ConformalThresholds:
    """Fit global (context_edges None or [0, 1]) or Mondrian thresholds on
    calibration OBJECTS.

    scores/proposed: object score S_j and whether the object has a candidate.
    context: per-object occlusion context o_hat (ignored for a global fit).
    """
    scores = np.asarray(scores, dtype=float)
    proposed = np.asarray(proposed, dtype=bool)
    if len(scores) == 0:
        raise ValueError("fit_thresholds needs at least one calibration object")
    edges = [0.0, 1.0] if not context_edges else [float(e) for e in context_edges]
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"context_edges must be strictly increasing, got {edges}")
    if len(edges) == 2 or context is None:
        edges = [0.0, 1.0]
        strata = np.zeros(len(scores), dtype=int)
    else:
        strata = stratum_indices(context, edges)
    n_strata = len(edges) - 1
    counts = np.bincount(strata, minlength=n_strata).tolist()
    mapping = merge_small_strata(counts, min_group_size)
    group = np.asarray(mapping)[strata]
    eff = _effective(scores, proposed)
    groups, calib_scores = [], []
    for g in range(max(mapping) + 1):
        m = group == g
        groups.append(miss_aware_quantile(scores[m], proposed[m], alpha))
        calib_scores.append(np.sort(eff[m]).tolist())
    return ConformalThresholds(alpha=alpha, score_name=score_name, context_edges=edges,
                               group_of_stratum=mapping, groups=groups, calib_scores=calib_scores)


def group_indices(context: np.ndarray | None, n: int, thr: ConformalThresholds) -> np.ndarray:
    if not thr.is_conditional and len(thr.context_edges) == 2:
        return np.zeros(n, dtype=int)
    if context is None:
        raise ValueError("these thresholds are occlusion-conditional; a context array is required")
    return np.asarray(thr.group_of_stratum)[stratum_indices(context, thr.context_edges)]


def keep_mask(scores: np.ndarray, context: np.ndarray | None, thr: ConformalThresholds) -> np.ndarray:
    """Which candidates the conformal rule keeps: score >= threshold of the
    candidate's own context group.
    """
    scores = np.asarray(scores, dtype=float)
    g = group_indices(context, len(scores), thr)
    t = np.array([grp.threshold for grp in thr.groups])[g]
    return scores >= t


def rank_transform(scores: np.ndarray, context: np.ndarray | None, thr: ConformalThresholds) -> np.ndarray:
    """Conformal rank of each candidate within its context group:

        r_k = (#{calibration objects in the group with S <= s_k} + 1) / (n_group + 1).

    Candidate k is kept by the level-alpha rule exactly when r_k > alpha, so
    sweeping a single threshold over r traces out the whole family of
    Mondrian rules. That makes r a POLICY SCORE that can be compared with raw
    confidence at any matched false-alarm budget.
    """
    scores = np.asarray(scores, dtype=float)
    g = group_indices(context, len(scores), thr)
    out = np.zeros(len(scores))
    for gi, calib in enumerate(thr.calib_scores):
        m = g == gi
        if not m.any():
            continue
        calib = np.asarray(calib, dtype=float)
        out[m] = (np.searchsorted(calib, scores[m], side="right") + 1) / (len(calib) + 1)
    return out


def coverage(scores: np.ndarray, proposed: np.ndarray, context: np.ndarray | None,
             thr: ConformalThresholds, weights: np.ndarray | None = None) -> dict:
    """Fraction of objects found by the rule (optionally weighted), overall
    and per context group. Objects without a candidate are never found.
    """
    scores = np.asarray(scores, dtype=float)
    proposed = np.asarray(proposed, dtype=bool)
    w = np.ones(len(scores)) if weights is None else np.asarray(weights, dtype=float)
    g = group_indices(context, len(scores), thr)
    found = proposed & keep_mask(scores, context, thr)
    per_group = {}
    for gi in range(len(thr.groups)):
        m = g == gi
        tot = float(w[m].sum())
        per_group[gi] = {"n": int(m.sum()), "coverage": float((w[m] * found[m]).sum() / tot) if tot > 0 else None}
    tot = float(w.sum())
    return {"coverage": float((w * found).sum() / tot) if tot > 0 else None,
            "n": int(len(scores)), "per_group": per_group}
