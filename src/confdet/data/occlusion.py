"""Occlusion severity (ground-truth side) and occlusion context (prediction
side). docs/theoretical_foundations.md sections 1 and 2.

Two closely related quantities live here, and keeping them apart is the
point of this module:

* ``occlusion severity``  o_j  -- computed from GROUND-TRUTH boxes. It is an
  analysis covariate: it tells us, after the fact, how occluded each annotated
  vehicle was. It exists only where annotations exist, so it can never be an
  input to a decision the deployed system makes.

* ``occlusion context``  o_hat_k -- computed from the DETECTOR'S OWN confident
  detections. It answers "how much of this candidate box sits behind a vehicle
  the detector is already sure about?" and is available at test time with no
  labels. Every decision rule in this project (Mondrian strata, the
  occlusion-aware score) is conditioned on this quantity, never on o_j.

Boxes are (x1, y1, x2, y2) in absolute pixels throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

ProxyKind = Literal["bottom_y", "area"]


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = -1

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def bottom_y(self) -> float:
        return self.y2


# ---------------------------------------------------------------------------
# Reference (pure-Python) implementation. Kept because it is the version the
# original unit tests pin down; the numpy implementation below is tested for
# exact agreement with it on random boxes (tests/test_occlusion.py).
# ---------------------------------------------------------------------------
def _is_nearer(candidate: Box, reference: Box, proxy: ProxyKind) -> bool:
    """Is `candidate` plausibly nearer to the camera than `reference`?

    bottom_y: on an oblique, downward-looking fixed camera an object whose
    bottom edge is lower in the frame stands nearer on the ground plane.
    area: larger box assumed nearer (fallback for near-frontal views).
    Ties are resolved as "not nearer", so two boxes with identical proxy
    value never occlude each other.
    """
    if proxy == "bottom_y":
        return candidate.bottom_y > reference.bottom_y
    if proxy == "area":
        return candidate.area > reference.area
    raise ValueError(f"Unknown occlusion proxy: {proxy!r}")


def _union_area(boxes: list[Box]) -> float:
    """Exact union area of a small set of rectangles (coordinate compression)."""
    if not boxes:
        return 0.0
    xs = sorted({b.x1 for b in boxes} | {b.x2 for b in boxes})
    ys = sorted({b.y1 for b in boxes} | {b.y2 for b in boxes})
    total = 0.0
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x1 <= x0:
            continue
        cx = (x0 + x1) / 2
        for j in range(len(ys) - 1):
            y0, y1 = ys[j], ys[j + 1]
            if y1 <= y0:
                continue
            cy = (y0 + y1) / 2
            if any(b.x1 <= cx <= b.x2 and b.y1 <= cy <= b.y2 for b in boxes):
                total += (x1 - x0) * (y1 - y0)
    return total


def compute_occlusion_severity(
    target: Box,
    other_boxes: Iterable[Box],
    proxy: ProxyKind = "bottom_y",
) -> float:
    """occlusion(target) = area(target ∩ union(nearer boxes)) / area(target)."""
    if target.area <= 0:
        return 0.0
    occluders = [b for b in other_boxes if _is_nearer(b, target, proxy)]
    clipped = []
    for b in occluders:
        ix1, iy1 = max(b.x1, target.x1), max(b.y1, target.y1)
        ix2, iy2 = min(b.x2, target.x2), min(b.y2, target.y2)
        if ix2 > ix1 and iy2 > iy1:
            clipped.append(Box(ix1, iy1, ix2, iy2))
    if not clipped:
        return 0.0
    return float(np.clip(_union_area(clipped) / target.area, 0.0, 1.0))


def compute_occlusion_for_image(boxes: list[Box], proxy: ProxyKind = "bottom_y") -> list[float]:
    """Severity of every box against all other boxes in the same image."""
    return [
        compute_occlusion_severity(b, [o for o in boxes if o is not b], proxy)
        for b in boxes
    ]


def bucket_index(severity: float, bucket_edges: list[float]) -> int:
    """Map a value in [0, 1] to its bucket. Edges must be sorted and span
    [0, 1]; bucket i is [edges[i], edges[i+1]) with the last bucket closed.
    """
    edges = np.asarray(bucket_edges)
    idx = int(np.searchsorted(edges, severity, side="right") - 1)
    return int(np.clip(idx, 0, len(edges) - 2))


def bucket_indices(values: np.ndarray, bucket_edges: list[float]) -> np.ndarray:
    """Vectorised `bucket_index`."""
    edges = np.asarray(bucket_edges, dtype=float)
    idx = np.searchsorted(edges, np.asarray(values, dtype=float), side="right") - 1
    return np.clip(idx, 0, len(edges) - 2).astype(int)


# ---------------------------------------------------------------------------
# Vectorised implementation (numpy). Same definitions, array in / array out.
# ---------------------------------------------------------------------------
def boxes_to_array(boxes: Iterable[Box]) -> np.ndarray:
    arr = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=float)
    return arr.reshape(-1, 4)


def box_areas(xyxy: np.ndarray) -> np.ndarray:
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    return np.clip(xyxy[:, 2] - xyxy[:, 0], 0, None) * np.clip(xyxy[:, 3] - xyxy[:, 1], 0, None)


def intersection_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection areas, shape (len(a), len(b))."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, shape (len(a), len(b))."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    inter = intersection_matrix(a, b)
    union = box_areas(a)[:, None] + box_areas(b)[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def _union_area_clipped(target: np.ndarray, occluders: np.ndarray) -> float:
    """Area of target ∩ union(occluders). `occluders` is (m, 4), m >= 1."""
    x1 = np.maximum(occluders[:, 0], target[0])
    y1 = np.maximum(occluders[:, 1], target[1])
    x2 = np.minimum(occluders[:, 2], target[2])
    y2 = np.minimum(occluders[:, 3], target[3])
    keep = (x2 > x1) & (y2 > y1)
    if not keep.any():
        return 0.0
    x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
    if len(x1) == 1:
        return float((x2[0] - x1[0]) * (y2[0] - y1[0]))
    xs = np.unique(np.concatenate([x1, x2]))
    ys = np.unique(np.concatenate([y1, y2]))
    cx = (xs[:-1] + xs[1:]) / 2
    cy = (ys[:-1] + ys[1:]) / 2
    in_x = (x1[:, None] <= cx[None, :]) & (cx[None, :] <= x2[:, None])   # (m, nx)
    in_y = (y1[:, None] <= cy[None, :]) & (cy[None, :] <= y2[:, None])   # (m, ny)
    covered = (in_x[:, :, None] & in_y[:, None, :]).any(axis=0)          # (nx, ny)
    cell = np.diff(xs)[:, None] * np.diff(ys)[None, :]
    return float((cell * covered).sum())


def _proxy_values(xyxy: np.ndarray, proxy: ProxyKind) -> np.ndarray:
    if proxy == "bottom_y":
        return xyxy[:, 3]
    if proxy == "area":
        return box_areas(xyxy)
    raise ValueError(f"Unknown occlusion proxy: {proxy!r}")


def covered_fraction(
    targets: np.ndarray,
    occluders: np.ndarray,
    proxy: ProxyKind = "bottom_y",
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    """For every target box: fraction of its area covered by the union of the
    occluder boxes that are nearer to the camera under `proxy`.

    allowed: optional boolean (n_targets, n_occluders) mask; an occluder is
    only considered for a target where the mask is True (used to exclude a
    box from occluding itself, and to exclude near-duplicates).
    """
    targets = np.asarray(targets, dtype=float).reshape(-1, 4)
    occluders = np.asarray(occluders, dtype=float).reshape(-1, 4)
    n, m = len(targets), len(occluders)
    out = np.zeros(n)
    if n == 0 or m == 0:
        return out
    t_area = box_areas(targets)
    inter = intersection_matrix(targets, occluders)
    nearer = _proxy_values(occluders, proxy)[None, :] > _proxy_values(targets, proxy)[:, None]
    usable = nearer & (inter > 0)
    if allowed is not None:
        usable &= np.asarray(allowed, dtype=bool)
    counts = usable.sum(axis=1)
    for i in np.nonzero((counts > 0) & (t_area > 0))[0]:
        if counts[i] == 1:
            covered = inter[i, usable[i]][0]
        else:
            covered = _union_area_clipped(targets[i], occluders[usable[i]])
        out[i] = min(1.0, max(0.0, covered / t_area[i]))
    return out


def occlusion_severity_array(gt_xyxy: np.ndarray, proxy: ProxyKind = "bottom_y") -> np.ndarray:
    """Ground-truth occlusion severity o_j for all boxes of ONE image."""
    gt_xyxy = np.asarray(gt_xyxy, dtype=float).reshape(-1, 4)
    n = len(gt_xyxy)
    if n < 2:
        return np.zeros(n)
    not_self = ~np.eye(n, dtype=bool)
    return covered_fraction(gt_xyxy, gt_xyxy, proxy, allowed=not_self)


def occlusion_context(
    cand_xyxy: np.ndarray,
    cand_conf: np.ndarray,
    context_conf: float,
    duplicate_iou: float = 0.5,
    proxy: ProxyKind = "bottom_y",
) -> tuple[np.ndarray, np.ndarray]:
    """Deployable occlusion context for every candidate of ONE image.

    Let D = {candidates with confidence >= context_conf} be the detections the
    detector is already confident about. For candidate k:

      dup_k   = max IoU(b_k, d) over d in D, d != k
      o_hat_k = area(b_k ∩ union of nearer d in D with d != k and
                                         IoU(b_k, d) < duplicate_iou) / area(b_k)

    The IoU exclusion keeps a second, lower-scored box on the SAME vehicle
    (a duplicate) from being read as "a vehicle hidden behind another one";
    duplicates are described by dup_k instead.

    Returns (o_hat, dup), both shape (n_candidates,). Uses only detector
    outputs, so it is available at test time.
    """
    cand_xyxy = np.asarray(cand_xyxy, dtype=float).reshape(-1, 4)
    cand_conf = np.asarray(cand_conf, dtype=float).reshape(-1)
    n = len(cand_xyxy)
    o_hat, dup = np.zeros(n), np.zeros(n)
    conf_idx = np.nonzero(cand_conf >= context_conf)[0]
    if n == 0 or len(conf_idx) == 0:
        return o_hat, dup
    ref = cand_xyxy[conf_idx]
    iou = iou_matrix(cand_xyxy, ref)
    not_self = np.arange(n)[:, None] != conf_idx[None, :]
    dup = np.where(not_self, iou, 0.0).max(axis=1)
    allowed = not_self & (iou < duplicate_iou)
    o_hat = covered_fraction(cand_xyxy, ref, proxy, allowed=allowed)
    return o_hat, dup


def occlusion_context_at_boxes(
    boxes_xyxy: np.ndarray,
    cand_xyxy: np.ndarray,
    cand_conf: np.ndarray,
    context_conf: float,
    proxy: ProxyKind = "bottom_y",
    duplicate_iou: float = 0.5,
) -> np.ndarray:
    """Occlusion context evaluated at arbitrary box locations. Used only for
    bookkeeping: a ground-truth object the detector proposed nothing for
    still has to be assigned to a context stratum when coverage is tallied.

    A confident detection that overlaps the box with IoU >= duplicate_iou is
    a detection OF THAT VEHICLE (under class-aware matching: with the wrong
    class), not something standing in front of it, and is excluded exactly as
    in `occlusion_context`. Without this exclusion every misnamed vehicle
    whose box edge happens to sit a pixel higher than its detection lands in
    the "heavily occluded" stratum -- which is what made that stratum look
    uncertifiable in the first real run (22% "unproposed" there, 5% elsewhere).
    """
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    cand_xyxy = np.asarray(cand_xyxy, dtype=float).reshape(-1, 4)
    cand_conf = np.asarray(cand_conf, dtype=float).reshape(-1)
    ref = cand_xyxy[cand_conf >= context_conf]
    if len(boxes_xyxy) == 0 or len(ref) == 0:
        return np.zeros(len(boxes_xyxy))
    allowed = iou_matrix(boxes_xyxy, ref) < duplicate_iou
    return covered_fraction(boxes_xyxy, ref, proxy, allowed=allowed)
