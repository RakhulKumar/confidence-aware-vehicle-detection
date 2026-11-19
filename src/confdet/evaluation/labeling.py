"""From cached predictions to analysis tables.
docs/theoretical_foundations.md section 3.

One matching pass per image, on the RAW candidate set, labels everything:

  * candidates are processed in descending confidence; each takes the still
    unmatched ground truth (same class, unless class-agnostic) it overlaps
    most, if that IoU >= iou_threshold. Matched -> true positive, otherwise
    false positive. This is the COCO evaluation protocol.
  * every ground-truth object j therefore has at most one candidate. Its
    OBJECT SCORE S_j is that candidate's confidence; an object nothing was
    proposed for is `unproposed` (S_j = -infinity in the theory, stored as
    proposed=False).

Because candidates are processed in confidence order, the matching of any
thresholded subset {confidence >= t} is exactly the prefix of this one
matching. So "object j is detected at threshold t" is simply S_j >= t, and
"candidate k is a false alarm at threshold t" is simply (k is FP and
confidence_k >= t). Every operating point, every policy and every metric in
this project is computed from these fixed labels; nothing is re-matched.

The tables are plain numpy arrays (one row per candidate, one row per ground
truth), which is what makes image-level bootstrap resampling cheap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from confdet.data.occlusion import (
    iou_matrix,
    occlusion_context,
    occlusion_context_at_boxes,
)
from confdet.evaluation.records import ImageRecord


@dataclass
class Tables:
    n_images: int
    image_ids: list[str]
    class_agnostic: bool
    iou_threshold: float
    context_conf: float
    # ---- candidates (one row per raw candidate kept for evaluation) ----
    c_img: np.ndarray        # image index
    c_conf: np.ndarray       # detector confidence
    c_cls: np.ndarray        # class id (after any class map)
    c_tp: np.ndarray         # bool: matched to a ground truth
    c_gt: np.ndarray         # row in the ground-truth table, -1 if false positive
    c_ctx: np.ndarray        # occlusion context o_hat (deployable)
    c_dup: np.ndarray        # max IoU with another confident detection (deployable)
    c_area: np.ndarray       # log(box area / image area) (deployable)
    # ---- ground truths (one row per annotated object kept for evaluation) ----
    g_img: np.ndarray
    g_cls: np.ndarray
    g_occ: np.ndarray        # occlusion severity o_j (annotation-derived; analysis only)
    g_proposed: np.ndarray   # bool: some raw candidate matched it
    g_score: np.ndarray      # S_j: matched candidate's confidence, 0.0 when unproposed
    g_cand: np.ndarray       # row in the candidate table, -1 when unproposed
    g_ctx: np.ndarray        # context stratum covariate: matched candidate's o_hat,
    #                          or o_hat evaluated at the ground-truth box when unproposed
    g_area: np.ndarray       # log(box area / image area)
    # class-agnostic localisation view of each object (RQ5). Candidates overlapping
    # object j (IoU >= thr, any class) are either its own match, FREE (matched to no
    # object), or a NEIGHBOUR's (matched to a different object).
    g_loc_score: np.ndarray  # highest confidence among own + free overlapping candidates (0 if none)
    g_loc_cls: np.ndarray    # class of that candidate (-1 if none)
    g_nbr_score: np.ndarray  # highest confidence among neighbours' boxes overlapping it (0 if none)
    ov_ptr: np.ndarray       # ragged: candidates overlapping object j are ov_*[ov_ptr[j]:ov_ptr[j+1]]
    ov_conf: np.ndarray
    ov_cls: np.ndarray
    ov_iou: np.ndarray
    # index of each kept candidate in its image's RAW candidate list (records order), so per-candidate
    # quantities computed on the raw records can be aligned after class-map and ignore-region filtering
    c_src: np.ndarray | None = None

    @property
    def n_gt(self) -> int:
        return len(self.g_img)

    @property
    def n_cand(self) -> int:
        return len(self.c_img)


def greedy_match(
    cand_xyxy: np.ndarray, cand_conf: np.ndarray, cand_cls: np.ndarray,
    gt_xyxy: np.ndarray, gt_cls: np.ndarray,
    iou_threshold: float = 0.5, class_agnostic: bool = False,
    iou: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """COCO-style greedy matching for one image.

    Returns (cand_to_gt, gt_to_cand), index arrays with -1 for "unmatched".
    """
    n_c, n_g = len(cand_conf), len(gt_cls)
    cand_to_gt = np.full(n_c, -1, dtype=np.int64)
    gt_to_cand = np.full(n_g, -1, dtype=np.int64)
    if n_c == 0 or n_g == 0:
        return cand_to_gt, gt_to_cand
    if iou is None:
        iou = iou_matrix(cand_xyxy, gt_xyxy)
    eligible = iou.copy()
    if not class_agnostic:
        eligible[np.asarray(cand_cls)[:, None] != np.asarray(gt_cls)[None, :]] = 0.0
    could_match = (eligible >= iou_threshold).any(axis=1)
    order = np.argsort(-np.asarray(cand_conf), kind="stable")
    free = np.ones(n_g, dtype=bool)
    for k in order:
        if not could_match[k]:
            continue
        row = np.where(free, eligible[k], 0.0)
        j = int(np.argmax(row))
        if row[j] >= iou_threshold and row[j] > 0:
            cand_to_gt[k] = j
            gt_to_cand[j] = k
            free[j] = False
            if not free.any():
                break
    return cand_to_gt, gt_to_cand


def _apply_map(cls: np.ndarray, class_map: np.ndarray | None) -> np.ndarray:
    if class_map is None:
        return cls.astype(np.int64)
    class_map = np.asarray(class_map, dtype=np.int64)
    out = np.full(len(cls), -1, dtype=np.int64)
    ok = (cls >= 0) & (cls < len(class_map))
    out[ok] = class_map[cls[ok]]
    return out


def build_tables(
    records: list[ImageRecord],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
    context_conf: float = 0.25,
    duplicate_iou: float = 0.5,
    proxy: str = "bottom_y",
    cand_class_map: np.ndarray | None = None,
    gt_class_map: np.ndarray | None = None,
) -> Tables:
    """Label every image and stack the results.

    cand_class_map / gt_class_map translate detector / dataset class ids into
    a common taxonomy (for cross-dataset evaluation). An id mapped to -1 has
    no counterpart:
      * such a ground-truth box becomes an IGNORE region -- it is not counted
        as an object, and a candidate sitting on it is not counted as a false
        alarm;
      * such a candidate is dropped (its class cannot be scored).
    Occlusion context is always computed from the full raw candidate set,
    before anything is dropped, because that is what the deployed system sees.
    """
    C = {k: [] for k in ("img", "conf", "cls", "tp", "gt", "ctx", "dup", "area", "src")}
    G = {k: [] for k in ("img", "cls", "occ", "prop", "score", "cand", "ctx", "area",
                         "loc_score", "loc_cls", "nbr_score")}
    ov_counts, ov_conf, ov_cls, ov_iou = [], [], [], []
    c_offset = g_offset = 0

    for i, r in enumerate(records):
        img_area = max(1.0, float(r.width) * float(r.height))
        ctx_all, dup_all = occlusion_context(r.cand_xyxy, r.cand_conf, context_conf, duplicate_iou, proxy)

        c_cls_all = _apply_map(r.cand_cls, cand_class_map)
        g_cls_all = _apply_map(r.gt_cls, gt_class_map)
        g_keep = g_cls_all >= 0
        ignore_xyxy = r.gt_xyxy[~g_keep]
        g_xyxy, g_cls, g_occ = r.gt_xyxy[g_keep], g_cls_all[g_keep], r.gt_occ[g_keep]

        c_keep = c_cls_all >= 0
        c_src = np.nonzero(c_keep)[0]
        c_xyxy, c_conf, c_cls = r.cand_xyxy[c_keep], r.cand_conf[c_keep], c_cls_all[c_keep]
        c_ctx, c_dup = ctx_all[c_keep], dup_all[c_keep]

        iou = iou_matrix(c_xyxy, g_xyxy)
        c2g, g2c = greedy_match(c_xyxy, c_conf, c_cls, g_xyxy, g_cls, iou_threshold, class_agnostic, iou=iou)

        if len(ignore_xyxy) and len(c_xyxy):
            on_ignore = (iou_matrix(c_xyxy, ignore_xyxy) >= iou_threshold).any(axis=1) & (c2g < 0)
            if on_ignore.any():
                keep = ~on_ignore
                remap = np.full(len(keep), -1, dtype=np.int64)
                remap[keep] = np.arange(int(keep.sum()))
                g2c = np.where(g2c >= 0, remap[np.clip(g2c, 0, None)], -1)
                c_xyxy, c_conf, c_cls = c_xyxy[keep], c_conf[keep], c_cls[keep]
                c_ctx, c_dup, c2g, iou = c_ctx[keep], c_dup[keep], c2g[keep], iou[keep]
                c_src = c_src[keep]

        n_c, n_g = len(c_conf), len(g_cls)
        c_areas = np.clip((c_xyxy[:, 2] - c_xyxy[:, 0]) * (c_xyxy[:, 3] - c_xyxy[:, 1]), 1e-6, None)
        g_areas = np.clip((g_xyxy[:, 2] - g_xyxy[:, 0]) * (g_xyxy[:, 3] - g_xyxy[:, 1]), 1e-6, None)

        C["img"].append(np.full(n_c, i)); C["conf"].append(c_conf.astype(float)); C["cls"].append(c_cls)
        C["tp"].append(c2g >= 0); C["gt"].append(np.where(c2g >= 0, c2g + g_offset, -1))
        C["ctx"].append(c_ctx); C["dup"].append(c_dup); C["area"].append(np.log(c_areas / img_area))
        C["src"].append(c_src)

        proposed = g2c >= 0
        score = np.where(proposed, c_conf[np.clip(g2c, 0, None)] if n_c else 0.0, 0.0).astype(float) \
            if n_g else np.zeros(0)
        g_ctx = np.zeros(n_g)
        if n_g:
            if proposed.any():
                g_ctx[proposed] = c_ctx[g2c[proposed]]
            if (~proposed).any():
                g_ctx[~proposed] = occlusion_context_at_boxes(
                    g_xyxy[~proposed], r.cand_xyxy, r.cand_conf, context_conf, proxy, duplicate_iou)

        loc_score, loc_cls = np.zeros(n_g), np.full(n_g, -1, dtype=np.int64)
        nbr_score = np.zeros(n_g)
        for j in range(n_g):
            if n_c:
                hits = np.nonzero(iou[:, j] >= iou_threshold)[0]
            else:
                hits = np.zeros(0, dtype=np.int64)
            ov_counts.append(len(hits))
            if len(hits):
                ov_conf.append(c_conf[hits].astype(float)); ov_cls.append(c_cls[hits]); ov_iou.append(iou[hits, j])
                neighbour = (c2g[hits] >= 0) & (c2g[hits] != j)
                if neighbour.any():
                    nbr_score[j] = float(c_conf[hits[neighbour]].max())
                mine = hits[~neighbour]
                if len(mine):
                    top = mine[np.argmax(c_conf[mine])]
                    loc_score[j], loc_cls[j] = float(c_conf[top]), int(c_cls[top])

        G["img"].append(np.full(n_g, i)); G["cls"].append(g_cls); G["occ"].append(g_occ.astype(float))
        G["prop"].append(proposed); G["score"].append(score)
        G["cand"].append(np.where(proposed, g2c + c_offset, -1)); G["ctx"].append(g_ctx)
        G["area"].append(np.log(g_areas / img_area))
        G["loc_score"].append(loc_score); G["loc_cls"].append(loc_cls); G["nbr_score"].append(nbr_score)

        c_offset += n_c
        g_offset += n_g

    def cat(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    return Tables(
        n_images=len(records), image_ids=[r.image_id for r in records],
        class_agnostic=class_agnostic, iou_threshold=iou_threshold, context_conf=context_conf,
        c_img=cat(C["img"], np.int64), c_conf=cat(C["conf"], float), c_cls=cat(C["cls"], np.int64),
        c_tp=cat(C["tp"], bool), c_gt=cat(C["gt"], np.int64), c_ctx=cat(C["ctx"], float),
        c_dup=cat(C["dup"], float), c_area=cat(C["area"], float),
        g_img=cat(G["img"], np.int64), g_cls=cat(G["cls"], np.int64), g_occ=cat(G["occ"], float),
        g_proposed=cat(G["prop"], bool), g_score=cat(G["score"], float), g_cand=cat(G["cand"], np.int64),
        g_ctx=cat(G["ctx"], float), g_area=cat(G["area"], float),
        g_loc_score=cat(G["loc_score"], float), g_loc_cls=cat(G["loc_cls"], np.int64),
        g_nbr_score=cat(G["nbr_score"], float),
        ov_ptr=np.cumsum([0] + ov_counts).astype(np.int64),
        ov_conf=cat(ov_conf, float), ov_cls=cat(ov_cls, np.int64), ov_iou=cat(ov_iou, float),
        c_src=cat(C["src"], np.int64),
    )


def class_map_array(names: list[str], mapping: dict[str, str | None] | None,
                    common: list[str]) -> np.ndarray:
    """Array a with a[class_id] = index in `common`, or -1 when the class is
    mapped to null / missing from `mapping`. With mapping=None the class
    names themselves must appear in `common`.
    """
    index = {c: i for i, c in enumerate(common)}
    out = np.full(len(names), -1, dtype=np.int64)
    for i, name in enumerate(names):
        target = name if mapping is None else mapping.get(name)
        if target is not None:
            if target not in index:
                raise ValueError(f"class {name!r} maps to {target!r}, which is not in the common taxonomy {common}")
            out[i] = index[target]
    return out
