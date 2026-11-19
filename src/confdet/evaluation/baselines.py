"""Phase 1 (v0.6) baselines from the conformal-detection and calibration
literature, computed on the same labelled tables as our own rules.

Every rule here is a single confidence threshold fitted on calibration images
and evaluated on test images. Notation: an object j is "found" at threshold t
iff it was proposed and its matched candidate's confidence S_j >= t
(labeling.build_tables; nothing is re-matched). Unproposed objects can never
be found: they count as misses in every risk below.

  crc_fnr_threshold        Conformal risk control on the per-image false-
                           negative rate (Angelopoulos et al. 2024; as used for
                           detection by Hulsman et al. 2024, arXiv:2412.20167,
                           and Andeol et al. 2023). Loss of image i at t:
                           FNR_i(t) = (objects of i not found at t) / (objects of i),
                           0 for an image without objects. Threshold: the
                           largest t with  n/(n+1) * mean_i FNR_i(t) + 1/(n+1) <= alpha.
                           Guarantee: E[FNR of a new image] <= alpha.
  seqcrc_confidence        SeqCRC confidence step (Andeol et al. 2025,
                           arXiv:2505.24038), losses as in the COD toolkit
                           (cods 0.2, od/loss.py): box-count-recall
                           max(0, (n_true - n_kept) / n_true) and box-count-
                           threshold 1[n_kept < n_true], where n_kept counts
                           ALL kept boxes (no spatial matching); risk corrected as
                           (sum of losses + B) / (n + 1) with B = 1 (cods'
                           FirstStepMonotonizingOptimizer). cods' own conformaliser
                           does not import in version 0.2 (a missing `Union`
                           import), so this re-implementation is used.
  pac_threshold            PAC prediction set over objects (Li et al. 2022,
                           arXiv:2204.07482): with n calibration objects, k* =
                           the largest k with BinomCDF(k; n, alpha) <= delta;
                           threshold = the largest t with misses(t) <= k*.
                           Guarantee: P(object miss rate > alpha) <= delta.
                           Objects are treated as exchangeable (images are not).
  gilg_features / fit_gilg Bivariate Beta calibration on (confidence, max IoU
                           with any higher-scoring detection of the image, class
                           ignored) -- the IoU-aware calibration of Gilg et al. 2023.
  laece0 / laace0          Localisation-aware ECE / ACE with TP threshold 0
                           (Kuzucu et al. 2024, arXiv:2405.20459): each kept
                           detection's target is the IoU of the object it matches
                           (greedy, any IoU > 0), 0 if none; 25 equal-width (ECE)
                           or equal-mass (ACE) confidence bins.
  lrp / olrp_threshold     Localisation-Recall-Precision error (Oksuz et al.)
                           at IoU tau = 0.5 and its optimal threshold, the LRP-
                           optimal operating point proposed by Kuzucu et al. 2024.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

from confdet.data.occlusion import bucket_indices, iou_matrix
from confdet.evaluation.labeling import greedy_match
from confdet.evaluation.stats import bootstrap_mean_ci, logistic_irls, percentile_ci, sigmoid


# ---------------------------------------------------------------------------
# thresholds with a guarantee
# ---------------------------------------------------------------------------
def _sweep(order_scores: np.ndarray):
    """Indices of the last element of each run of equal scores in a descending array:
    a threshold t keeps every element with score >= t, so risks are evaluated there."""
    s = order_scores
    return np.nonzero(np.r_[s[1:] != s[:-1], True])[0] if len(s) else np.zeros(0, int)


def per_image_fnr(found: np.ndarray, img: np.ndarray, n_images: int) -> np.ndarray:
    """FNR of every image (0 for an image with no objects)."""
    n = np.bincount(img, minlength=n_images).astype(float)
    miss = np.bincount(img, weights=(~found).astype(float), minlength=n_images)
    return np.divide(miss, n, out=np.zeros(n_images), where=n > 0)


def _first_admissible(scores_desc, total_after, n: int, alpha: float, B: float):
    """total_after[k]: summed loss once the first k+1 elements (descending) are kept.
    Returns (threshold, total) for the LARGEST threshold with (total + B)/(n + 1) <= alpha."""
    for k in _sweep(scores_desc):
        if (total_after[k] + B) / (n + 1) <= alpha + 1e-12:     # tolerance: floating-point sums at the boundary
            return float(scores_desc[k]), float(total_after[k])
    return None, None


def crc_fnr_threshold(g_score, g_proposed, g_img, n_images: int, alpha: float, B: float = 1.0) -> dict:
    """Largest threshold whose CRC bound on the per-image FNR is <= alpha:
    n/(n+1) * mean FNR + B/(n+1) <= alpha  <=>  (sum_i FNR_i + B)/(n+1) <= alpha."""
    g_score, g_proposed, g_img = np.asarray(g_score, float), np.asarray(g_proposed, bool), np.asarray(g_img)
    n = int(n_images)
    n_obj = np.bincount(g_img, minlength=n).astype(float)
    start = float((n_obj > 0).sum())                       # nothing kept: every image with objects has FNR 1
    idx = np.nonzero(g_proposed)[0]
    order = idx[np.argsort(-g_score[idx], kind="stable")]
    total = start - np.cumsum(1.0 / n_obj[g_img[order]])
    t, tot = _first_admissible(g_score[order], total, n, alpha, B)
    floor = total[-1] if len(total) else start
    return {"method": "crc_fnr", "alpha": alpha, "n_images": n, "achievable": t is not None,
            "min_attainable_bound": float((floor + B) / (n + 1)),
            **({"threshold": t, "calib_risk": tot / n, "bound": (tot + B) / (n + 1)} if t is not None
               else {"threshold": None})}


def seqcrc_confidence(g_img, n_images: int, c_img, c_conf, alpha: float, loss: str = "box_count_recall",
                      B: float = 1.0) -> dict:
    """SeqCRC confidence step with cods' box-count losses (no spatial matching)."""
    if loss not in ("box_count_recall", "box_count_threshold"):
        raise ValueError(loss)
    n = int(n_images)
    n_true = np.bincount(np.asarray(g_img), minlength=n).astype(float)
    c_img, c_conf = np.asarray(c_img), np.asarray(c_conf, float)
    order = np.argsort(-c_conf, kind="stable")
    im = c_img[order]
    # k-th kept box of an image (1-based) among that image's boxes, in descending confidence
    rank = np.zeros(len(order), int)
    seen = np.zeros(n, int)
    for k, i in enumerate(im):
        seen[i] += 1
        rank[k] = seen[i]
    useful = rank <= n_true[im]                            # boxes beyond n_true do not lower the loss
    if loss == "box_count_recall":
        start = float((n_true > 0).sum())
        dec = np.where(useful, 1.0 / np.maximum(n_true[im], 1), 0.0)
    else:
        start = float((n_true > 0).sum())
        dec = (useful & (rank == n_true[im])).astype(float)  # loss drops to 0 when the count is reached
    total = start - np.cumsum(dec)
    t, tot = _first_admissible(c_conf[order], total, n, alpha, B)
    return {"method": f"seqcrc_{loss}", "alpha": alpha, "n_images": n, "achievable": t is not None,
            "route": "re-implementation of the cods 0.2 confidence step (cods.od.cp does not import)",
            **({"threshold": t, "calib_risk": tot / n, "bound": (tot + B) / (n + 1)} if t is not None
               else {"threshold": None})}


def pac_k_star(n: int, alpha: float, delta: float) -> int:
    """Largest number of misses k with P(Binomial(n, alpha) <= k) <= delta; -1 if none."""
    ks = np.arange(n + 1)
    ok = binom.cdf(ks, n, alpha) <= delta
    return int(ks[ok].max()) if ok.any() else -1


def pac_threshold(g_score, g_proposed, alpha: float, delta: float = 0.05) -> dict:
    g_score, g_proposed = np.asarray(g_score, float), np.asarray(g_proposed, bool)
    n = len(g_score)
    k = pac_k_star(n, alpha, delta)
    idx = np.nonzero(g_proposed)[0]
    order = idx[np.argsort(-g_score[idx], kind="stable")]
    misses = n - np.arange(1, len(order) + 1)
    t = None
    if k >= 0:
        for j in _sweep(g_score[order]):
            if misses[j] <= k:
                t = float(g_score[order][j])
                break
    return {"method": "pac", "alpha": alpha, "delta": delta, "n_objects": n, "k_star": k,
            "achievable": t is not None,
            **({"threshold": t, "calib_misses": int(misses[j])} if t is not None else {"threshold": None})}


# ---------------------------------------------------------------------------
# per-candidate geometry from the records (aligned with the class-agnostic tables)
# ---------------------------------------------------------------------------
def candidate_geometry(records, iou_threshold: float = 0.5) -> dict:
    """For every candidate, in the order build_tables(class_agnostic=True) uses
    for a dataset without class maps:
      match_iou   IoU with the object it matches at iou_threshold (0 if none)
      tp          matched at iou_threshold (must equal tables.c_tp)
      la_target   IoU of the object it matches under greedy matching at ANY IoU > 0
                  (the LaECE_0 target; 0 if none)
      u_higher    max IoU with any higher-scoring candidate of the image (Gilg's feature)
    """
    out = {k: [] for k in ("match_iou", "tp", "la_target", "u_higher")}
    for r in records:
        c, g = np.asarray(r.cand_xyxy, float).reshape(-1, 4), np.asarray(r.gt_xyxy, float).reshape(-1, 4)
        conf = np.asarray(r.cand_conf, float)
        n_c = len(conf)
        iou = iou_matrix(c, g) if n_c and len(g) else np.zeros((n_c, len(g)))
        c2g, _ = greedy_match(c, conf, np.zeros(n_c, int), g, np.zeros(len(g), int), iou_threshold, True, iou=iou)
        m_iou = np.where(c2g >= 0, iou[np.arange(n_c), np.clip(c2g, 0, None)] if len(g) else 0.0, 0.0)
        c2g0, _ = greedy_match(c, conf, np.zeros(n_c, int), g, np.zeros(len(g), int), 1e-12, True, iou=iou)
        la = np.where(c2g0 >= 0, iou[np.arange(n_c), np.clip(c2g0, 0, None)] if len(g) else 0.0, 0.0)
        u = np.zeros(n_c)
        if n_c > 1:
            cc = iou_matrix(c, c)
            higher = conf[None, :] > conf[:, None]
            u = np.where(higher, cc, 0.0).max(axis=1)
        out["match_iou"].append(m_iou); out["tp"].append(c2g >= 0)
        out["la_target"].append(la); out["u_higher"].append(u)
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Gilg et al.: bivariate Beta calibration on (confidence, IoU with higher-scoring box)
# ---------------------------------------------------------------------------
def _beta_features(s, u, eps: float = 1e-4) -> np.ndarray:
    s = np.clip(np.asarray(s, float), eps, 1 - eps)
    u = np.clip(np.asarray(u, float), eps, 1 - eps)
    return np.column_stack([np.log(s), -np.log(1 - s), np.log(u), -np.log(1 - u)])


def fit_gilg(conf, u, is_tp, l2: float = 1e-4) -> dict:
    X = _beta_features(conf, u)
    mean, std = X.mean(axis=0), np.where(X.std(axis=0) < 1e-9, 1.0, X.std(axis=0))
    beta = logistic_irls(np.column_stack([np.ones(len(X)), (X - mean) / std]), np.asarray(is_tp, float), l2=l2)
    return {"features": ["ln s", "-ln(1-s)", "ln u", "-ln(1-u)"], "mean": mean.tolist(), "std": std.tolist(),
            "coef": beta.tolist(), "n_fit": int(len(X))}


def predict_gilg(model: dict, conf, u) -> np.ndarray:
    X = (_beta_features(conf, u) - np.asarray(model["mean"])) / np.asarray(model["std"])
    return sigmoid(model["coef"][0] + X @ np.asarray(model["coef"][1:]))


# ---------------------------------------------------------------------------
# Kuzucu et al. 2024: LaECE_0, LaACE_0, LRP and the LRP-optimal threshold
# ---------------------------------------------------------------------------
def _binned_gap(conf, target, edges_or_count, adaptive: bool) -> float:
    conf, target = np.asarray(conf, float), np.asarray(target, float)
    if len(conf) == 0:
        return 0.0
    if adaptive:
        order = np.argsort(conf, kind="stable")
        bins = np.array_split(order, min(int(edges_or_count), len(conf)))
    else:
        edges = np.linspace(0, 1, int(edges_or_count) + 1)
        idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)
        bins = [np.nonzero(idx == b)[0] for b in range(len(edges) - 1)]
    n = len(conf)
    return float(sum(len(b) / n * abs(conf[b].mean() - target[b].mean()) for b in bins if len(b)))


def laece0(conf, la_target, n_bins: int = 25) -> float:
    return _binned_gap(conf, la_target, n_bins, adaptive=False)


def laace0(conf, la_target, n_bins: int = 25) -> float:
    return _binned_gap(conf, la_target, n_bins, adaptive=True)


def lrp(tp_iou: np.ndarray, n_fp: float, n_fn: float, tau: float = 0.5) -> float:
    """LRP = (sum_TP (1 - IoU)/(1 - tau) + N_FP + N_FN) / (N_TP + N_FP + N_FN); 1 if nothing at all."""
    tp_iou = np.asarray(tp_iou, float)
    denom = len(tp_iou) + n_fp + n_fn
    return float(((1 - tp_iou) / (1 - tau)).sum() + n_fp + n_fn) / denom if denom > 0 else 1.0


def lrp_at(tables, geo: dict, t: float, tau: float = 0.5, w_img=None) -> dict:
    """LRP and its components at confidence threshold t (image weights for the bootstrap)."""
    w = np.ones(tables.n_images) if w_img is None else np.asarray(w_img, float)
    wc, wg = w[tables.c_img], w[tables.g_img]
    keep = tables.c_conf >= t
    tp = keep & tables.c_tp
    n_tp, n_fp = float(wc[tp].sum()), float(wc[keep & ~tables.c_tp].sum())
    n_fn = float(wg.sum()) - n_tp
    loc = float((wc[tp] * (1 - geo["match_iou"][tp]) / (1 - tau)).sum())
    denom = n_tp + n_fp + n_fn
    return {"lrp": (loc + n_fp + n_fn) / denom if denom > 0 else 1.0, "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn,
            "lrp_loc": loc / n_tp * (1 - tau) if n_tp else None}


def olrp_threshold(tables, geo: dict, grid=None, tau: float = 0.5) -> dict:
    grid = np.round(np.arange(0.01, 0.991, 0.01), 2) if grid is None else np.asarray(grid)
    vals = [lrp_at(tables, geo, float(t), tau)["lrp"] for t in grid]
    i = int(np.argmin(vals))
    return {"threshold": float(grid[i]), "olrp": float(vals[i])}


# ---------------------------------------------------------------------------
# evaluating any single-threshold rule on test, our way
# ---------------------------------------------------------------------------
def threshold_rule_report(tables, t: float | None, std_threshold: float, bucket_edges, context_edges,
                          boot_w: np.ndarray, min_bucket_n: int = 30) -> dict:
    """A single-threshold rule: object found iff proposed and its score >= t; flagged
    candidates are those with t <= confidence < the operating threshold."""
    if t is None:
        return {"threshold": None, "achievable": False}
    found = tables.g_proposed & (tables.g_score >= t)
    flagged = (tables.c_conf >= t) & (tables.c_conf < std_threshold)
    return {"threshold": float(t), "achievable": True,
            **report_from_found(tables, found, flagged, std_threshold, bucket_edges, context_edges, boot_w, min_bucket_n)}


def found_for_rule(tables, thr, cand_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(found per object, kept per candidate) for one of our fitted conformal rules
    (uq/conformal.ConformalThresholds on `cand_score`), exactly as rq2_recovery.conformal_report."""
    from confdet.uq.conformal import group_indices, keep_mask
    cand_score = np.asarray(cand_score, float)
    conditional = len(thr.context_edges) > 2
    kept = keep_mask(cand_score, tables.c_ctx if conditional else None, thr)
    obj = np.where(tables.g_proposed, cand_score[np.clip(tables.g_cand, 0, None)] if tables.n_cand else 0.0, 0.0)
    g_group = group_indices(tables.g_ctx if conditional else None, tables.n_gt, thr)
    t = np.array([g.threshold for g in thr.groups])[g_group] if tables.n_gt else np.zeros(0)
    return tables.g_proposed & (obj >= t), kept


def report_from_found(tables, found: np.ndarray, flagged: np.ndarray, std_threshold: float, bucket_edges,
                      context_edges, boot_w: np.ndarray, min_bucket_n: int = 30) -> dict:
    """Object recall of the rule alone and of the system (normal output + flagged
    tier), per-image FNR, flags per image, and coverage by annotated occlusion
    bucket (o_j) and by occlusion-context stratum (o_hat), with bootstrap intervals."""
    img = tables.g_img
    n_img = tables.n_images
    found_std = tables.g_proposed & (tables.g_score >= std_threshold)
    system = found | found_std
    fnr = per_image_fnr(found, img, n_img)
    has_obj = np.bincount(img, minlength=n_img) > 0
    fnr_reps = [(bw[has_obj] @ fnr[has_obj]) / bw[has_obj].sum() for bw in boot_w]

    def strata(values, edges):
        b = bucket_indices(values, edges)
        rows = []
        for k in range(len(edges) - 1):
            m = b == k
            nk = int(m.sum())
            rows.append({"lo": float(edges[k]), "hi": float(edges[k + 1]), "n": nk, "reliable": nk >= min_bucket_n,
                         "coverage": float(found[m].mean()) if nk else None,
                         "coverage_ci": bootstrap_mean_ci(found[m].astype(float), img[m], boot_w) if nk else None,
                         "system_coverage": float(system[m].mean()) if nk else None})
        return rows

    return {"coverage": float(found.mean()), "coverage_ci": bootstrap_mean_ci(found.astype(float), img, boot_w),
            "system_recall": float(system.mean()),
            "mean_image_fnr": float(fnr[has_obj].mean()), "mean_image_fnr_ci": percentile_ci(fnr_reps),
            "flags_per_image": float(flagged.sum() / max(n_img, 1)),
            "flagged_false_per_image": float((flagged & ~tables.c_tp).sum() / max(n_img, 1)),
            "by_annotated_occlusion": strata(tables.g_occ, bucket_edges),
            "by_occlusion_context": strata(tables.g_ctx, context_edges)}


def subset_images(tables, image_mask: np.ndarray) -> dict:
    """Object and candidate arrays restricted to some images, image ids renumbered 0..m-1."""
    image_mask = np.asarray(image_mask, bool)
    new_id = np.full(tables.n_images, -1)
    new_id[image_mask] = np.arange(int(image_mask.sum()))
    g, c = image_mask[tables.g_img], image_mask[tables.c_img]
    return {"n_images": int(image_mask.sum()), "g_score": tables.g_score[g], "g_proposed": tables.g_proposed[g],
            "g_img": new_id[tables.g_img[g]], "c_img": new_id[tables.c_img[c]], "c_conf": tables.c_conf[c],
            "g_mask": g, "c_mask": c}


def bootstrap_rows(values_fn, img: np.ndarray, boot_w: np.ndarray) -> tuple[float, float]:
    """Percentile interval of a statistic of per-row arrays under image resampling:
    each replicate repeats every row as often as its image was drawn."""
    reps = []
    for bw in boot_w:
        rep = np.repeat(np.arange(len(img)), bw[img].astype(int))
        reps.append(values_fn(rep))
    return percentile_ci(reps)


def align_to_tables(per_raw_candidate: np.ndarray, records, tables) -> np.ndarray:
    """Pick, for every candidate row of `tables`, the value computed on the raw records' candidates
    (concatenated in records order), using tables.c_src (robust to class-map and ignore filtering)."""
    offsets = np.cumsum([0] + [len(r.cand_conf) for r in records])
    return np.asarray(per_raw_candidate)[offsets[tables.c_img] + tables.c_src]
