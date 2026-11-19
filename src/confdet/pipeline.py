"""Glue between configs/experiment.yaml and the library, shared by scripts/.

Kept out of the scripts so the whole flow -- predictions -> tables ->
post-hoc layer -> RQ bundles -- is importable and is exercised end to end by
scripts/00_selftest.py without a GPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from confdet.calibration.detection_ece import operating_point
from confdet.data.occlusion import occlusion_severity_array
from confdet.data.splits import processed_info, role_images
from confdet.data.yolo_dataset import image_size, read_label_file
from confdet.evaluation.detection_metrics import detection_summary, usable_for_calibration_analysis
from confdet.evaluation.labeling import Tables, build_tables, class_map_array
from confdet.evaluation.records import ImageRecord, RecordSet
from confdet.evaluation.rq2_recovery import conformal_report
from confdet.evaluation.simulate import stable_seed
from confdet.uq.calibrator import CandidateCalibrator, fit_calibrator
from confdet.uq.conformal import ConformalThresholds, fit_thresholds

RULES = ("global_confidence", "mondrian_confidence", "global_occlusion_aware")
# Feature set of the fitted candidate score when posthoc.score_features is not given. The score
# keeps the result key "occlusion_aware" (and the rule "global_occlusion_aware") whatever its features.
DEFAULT_SCORE_FEATURES = "full_strict"
SCORE_LABELS = {"full_strict": "duplicate- and context-aware score",
                "occlusion_aware": "occlusion-aware score (z, ctx, dup)",
                "z_dup": "duplicate-aware score", "z_ctx": "context-aware score", "confidence": "Platt score"}


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------
def collect_records(detector, dataset: str, role: str, proxy: str = "bottom_y",
                    processed_root: str | Path = "data/processed", logger=None,
                    limit: int | None = None) -> RecordSet:
    """Run the frozen detector over one role of a materialised dataset."""
    info = processed_info(dataset, processed_root)
    pairs = role_images(dataset, role, processed_root)
    if limit:
        pairs = pairs[:limit]
    records = []
    for i, (image_path, label_path) in enumerate(pairs):
        w, h = image_size(image_path)
        rows, _ = read_label_file(label_path, "bbox")
        gt_xyxy = np.array([[r[1] * w, r[2] * h, r[3] * w, r[4] * h] for r in rows], dtype=float).reshape(-1, 4)
        gt_cls = np.array([r[0] for r in rows], dtype=int)
        occ = occlusion_severity_array(gt_xyxy, proxy=proxy)
        xyxy, conf, cls = detector.predict(image_path)
        records.append(ImageRecord(image_path.name, w, h, gt_xyxy, gt_cls, occ, xyxy, conf, cls))
        if logger and (i + 1) % 500 == 0:
            logger.info(f"    {i + 1}/{len(pairs)} images")
    return RecordSet(records, {"dataset": dataset, "role": role, "dataset_names": info.names,
                               "detector_names": list(detector.names), "occlusion_proxy": proxy})


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def max_f1_threshold(t: Tables, grid: np.ndarray | None = None) -> tuple[float, float]:
    """Confidence threshold maximising F1 on the given (calibration) tables,
    searched on a 0.01 grid. Returns (threshold, F1 there).

    Why a per-detector operating point: at one fixed threshold (0.25) the
    first real run had RT-DETR emitting 9 false alarms per image and the
    YOLO models 3.4, so every "which detector is more reliable" comparison
    was also a comparison of two different operating regimes.
    """
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else np.asarray(grid, dtype=float)
    if t.n_gt == 0 or t.n_cand == 0:
        return 0.25, 0.0
    order = np.argsort(-t.c_conf, kind="stable")
    conf, tp = t.c_conf[order], t.c_tp[order].astype(float)
    ctp = np.concatenate([[0.0], np.cumsum(tp)])
    k = np.searchsorted(-conf, -grid, side="right")          # number of candidates with conf >= threshold
    n_tp = ctp[k]
    f1 = 2 * n_tp / np.maximum(k + t.n_gt, 1)
    best = int(np.argmax(f1))
    return float(grid[best]), float(f1[best])


def tables_from_records(cfg: dict, rs: RecordSet, mode: str = "native", common: bool = False,
                        context_conf: float | None = None) -> Tables:
    """Label a record set under a matching mode (docs section 3):

      class_agnostic  a vehicle counts as found if ANY box overlaps it. This is
                      the notion "silently missed vehicle" is about, and the
                      primary mode for RQ1-RQ3 and the conformal layer.
      coarse          detector and dataset classes are both mapped, by name,
                      into evaluation.common_classes (e.g. Hatchback/Sedan/SUV
                      -> car) before class-aware matching.
      native          class-aware matching on the dataset's own fine-grained
                      classes. Only possible on the dataset the detector was
                      trained on; on any other dataset it falls back to coarse,
                      because class indices of two taxonomies are never compared.

    common=True also applies the class maps to the development dataset (so
    unmapped classes are ignored there too), which makes its numbers
    comparable with a transfer dataset's.
    context_conf: confidence at/above which a detection defines occlusion
    context; pass the detector's operating threshold. Falls back to
    posthoc.context_conf, then to a numeric evaluation.standard_operating_threshold.
    """
    if mode not in ("class_agnostic", "coarse", "native"):
        raise ValueError(f"unknown matching mode {mode!r}")
    ev, ph = cfg["evaluation"], cfg["posthoc"]
    dataset = rs.meta["dataset"]
    dev = cfg["experiment"]["development"]
    if context_conf is None:
        context_conf = ph.get("context_conf")
    if context_conf is None:
        spec = ev["standard_operating_threshold"]
        if isinstance(spec, str):
            raise ValueError("context_conf must be passed when standard_operating_threshold is 'max_f1'")
        context_conf = float(spec)
    cand_map = gt_map = None
    if dataset != dev or common or mode == "coarse":
        common_classes = ev.get("common_classes")
        if not common_classes:
            raise KeyError("evaluation.common_classes is required for coarse or cross-dataset evaluation")
        dev_map, ds_map = cfg["datasets"][dev].get("class_map"), cfg["datasets"][dataset].get("class_map")
        if dev_map is None or ds_map is None:
            raise KeyError(f"datasets.{dev}.class_map and datasets.{dataset}.class_map are both required "
                           f"for coarse or cross-dataset evaluation")
        for label, names, mapping in (("detector", rs.meta["detector_names"], dev_map),
                                      (dataset, rs.meta["dataset_names"], ds_map)):
            missing = [n for n in names if n not in mapping]
            if missing:
                raise KeyError(f"class_map for {label} has no entry for {missing}; map each class to a "
                               f"common class or to null")
        cand_map = class_map_array(rs.meta["detector_names"], dev_map, common_classes)
        gt_map = class_map_array(rs.meta["dataset_names"], ds_map, common_classes)
    return build_tables(
        rs.records,
        iou_threshold=float(ev.get("iou_match_threshold", 0.5)),
        class_agnostic=(mode == "class_agnostic"),
        context_conf=float(context_conf),
        duplicate_iou=float(ph.get("duplicate_iou", 0.5)),
        proxy=cfg["occlusion"].get("proxy", "bottom_y"),
        cand_class_map=cand_map, gt_class_map=gt_map,
    )


def transfer_mode(mode: str) -> str:
    """Matching mode used on a dataset with a different taxonomy."""
    return "class_agnostic" if mode == "class_agnostic" else "coarse"


def calibration_tables(cfg: dict, rs: RecordSet, mode: str) -> tuple[Tables, float, float]:
    """Calibration tables plus the operating threshold and context threshold
    they imply. Returns (tables, tau, context_conf).

    evaluation.standard_operating_threshold: a number, or "max_f1" to pick
    the F1-optimal threshold of THIS detector on the calib role.
    posthoc.context_conf: a number, or null to use tau.
    """
    spec = cfg["evaluation"]["standard_operating_threshold"]
    ctx = cfg["posthoc"].get("context_conf")
    if isinstance(spec, str):
        if spec != "max_f1":
            raise ValueError(f"standard_operating_threshold must be a number or 'max_f1', got {spec!r}")
        # matching (and therefore F1) does not depend on the context threshold
        tau, _ = max_f1_threshold(tables_from_records(cfg, rs, mode, context_conf=0.25))
    else:
        tau = float(spec)
    ctx = tau if ctx is None else float(ctx)
    return tables_from_records(cfg, rs, mode, context_conf=ctx), tau, ctx


# ---------------------------------------------------------------------------
# post-hoc layer
# ---------------------------------------------------------------------------
def calibrator_candidate_rows(calib: Tables, cfg: dict, seed: int) -> np.ndarray:
    """Boolean per calib candidate: True for the candidates fit_posthoc fits
    the candidate score on (scripts/13 fits its ablation sets on the same rows).
    """
    half = _calibrator_half(calib.image_ids, float(cfg["posthoc"].get("calibrator_fraction", 0.5)), seed)
    return half[calib.c_img] if calib.n_cand else np.zeros(0, dtype=bool)


def _calibrator_half(image_ids: list[str], fraction: float, seed: int) -> np.ndarray:
    """Boolean per image: True -> used to fit the candidate score, False ->
    used to fit conformal thresholds. The two never share an image, so the
    thresholds are computed on scores from a function fitted elsewhere, as
    split conformal prediction requires.
    """
    return np.array([(stable_seed(i, seed) % 10000) < fraction * 10000 for i in image_ids], dtype=bool)


def fit_posthoc(cfg: dict, calib: Tables, seed: int = 0, tau: float | None = None,
                mode: str | None = None) -> dict:
    """Fit the candidate score and all conformal thresholds on the calib role.

    tau: the detector's operating threshold these fits go with (recorded so
    every later stage uses the same one); defaults to the context threshold
    the tables were built with.
    """
    ph = cfg["posthoc"]
    tau = float(calib.context_conf if tau is None else tau)
    mode = mode or ("class_agnostic" if calib.class_agnostic else "native")
    alphas = sorted({float(a) for a in list(ph.get("alpha_grid", [])) + [ph["alpha"]]})
    half = _calibrator_half(calib.image_ids, float(ph.get("calibrator_fraction", 0.5)), seed)
    c_fit = half[calib.c_img] if calib.n_cand else np.zeros(0, dtype=bool)
    g_conf = ~half[calib.g_img] if calib.n_gt else np.zeros(0, dtype=bool)
    out = {"meta": {"n_calib_images": calib.n_images, "n_images_for_score": int(half.sum()),
                    "n_images_for_thresholds": int((~half).sum()),
                    "n_objects_for_thresholds": int(g_conf.sum()),
                    "context_edges": list(ph["context_edges"]), "alpha": float(ph["alpha"]),
                    "matching": mode, "standard_threshold": tau,
                    "context_conf": float(calib.context_conf)},
           "calibrator": None, "platt": None, "notes": [], "thresholds": {}}
    if g_conf.sum() == 0:
        raise ValueError(f"the calib role leaves no annotated vehicles to fit thresholds on "
                         f"({calib.n_images} images, {calib.n_gt} vehicles); it needs more images")

    calibrator = None
    dup_iou = float(ph.get("duplicate_iou", 0.5))
    features = str(ph.get("score_features", DEFAULT_SCORE_FEATURES))
    out["meta"]["score_features"] = features
    try:
        calibrator = fit_calibrator(calib.c_conf[c_fit], calib.c_ctx[c_fit], calib.c_dup[c_fit],
                                    calib.c_tp[c_fit], features, duplicate_iou=dup_iou)
        out["calibrator"] = calibrator.to_dict()
        out["platt"] = fit_calibrator(calib.c_conf[c_fit], calib.c_ctx[c_fit], calib.c_dup[c_fit],
                                      calib.c_tp[c_fit], "confidence", duplicate_iou=dup_iou).to_dict()
    except ValueError as e:
        out["notes"].append(f"occlusion-aware score not fitted: {e}")

    score, prop, ctx = calib.g_score[g_conf], calib.g_proposed[g_conf], calib.g_ctx[g_conf]
    obj_aware = None
    if calibrator is not None:
        cand_aware = calibrator.predict(calib.c_conf, calib.c_ctx, calib.c_dup)
        obj_aware = np.where(calib.g_proposed, cand_aware[np.clip(calib.g_cand, 0, None)]
                             if calib.n_cand else 0.0, 0.0)[g_conf]
    for a in alphas:
        entry = {
            "global_confidence": fit_thresholds(score, prop, None, a).to_dict(),
            "mondrian_confidence": fit_thresholds(
                score, prop, ctx, a, context_edges=list(ph["context_edges"]),
                min_group_size=int(ph.get("min_group_size", 100))).to_dict(),
        }
        if obj_aware is not None:
            entry["global_occlusion_aware"] = fit_thresholds(
                obj_aware, prop, None, a, score_name="occlusion_aware").to_dict()
        out["thresholds"][f"{a:g}"] = entry
    return out


class PostHoc:
    """Loaded post-hoc layer (outputs/<exp>/<detector>/calibration.json)."""

    def __init__(self, d: dict):
        self.raw = d
        self.calibrator = CandidateCalibrator.from_dict(d["calibrator"]) if d.get("calibrator") else None
        self.alpha = float(d["meta"]["alpha"])
        self.tau = float(d["meta"]["standard_threshold"])           # the detector's operating threshold
        self.context_conf = float(d["meta"]["context_conf"])
        self.mode = d["meta"].get("matching", "native")
        self.thresholds = {a: {name: ConformalThresholds.from_dict(t) for name, t in rules.items()}
                           for a, rules in d["thresholds"].items()}

    @property
    def alphas(self) -> list[float]:
        return sorted(float(a) for a in self.thresholds)

    def rules(self, alpha: float | None = None) -> dict[str, ConformalThresholds]:
        return self.thresholds[f"{self.alpha if alpha is None else alpha:g}"]

    def candidate_scores(self, t: Tables) -> dict[str, np.ndarray]:
        """Per-candidate score each rule's thresholds apply to."""
        out = {"global_confidence": t.c_conf, "mondrian_confidence": t.c_conf}
        if self.calibrator is not None:
            out["global_occlusion_aware"] = self.calibrator.predict(t.c_conf, t.c_ctx, t.c_dup)
        return out

    def rule_inputs(self, t: Tables, alpha: float | None = None):
        scores = self.candidate_scores(t)
        return {name: (thr, scores[name]) for name, thr in self.rules(alpha).items() if name in scores}


def validity_curve(posthoc: PostHoc, t: Tables, std_threshold: float, bucket_edges, boot_w) -> list[dict]:
    """Promised vs delivered recall for every alpha that was calibrated."""
    rows = []
    for a in posthoc.alphas:
        rep = conformal_report(t, posthoc.rule_inputs(t, a), std_threshold, bucket_edges, boot_w)
        row = {"alpha": a, "target_coverage": 1 - a}
        for name in RULES:
            if name in rep:
                row[name] = {"coverage": rep[name]["coverage"], "coverage_ci": rep[name]["coverage_ci"],
                             "all_groups_achievable": rep[name]["all_groups_achievable"],
                             "flagged_per_image": rep[name]["flagged_per_image"],
                             "system_recall": rep[name]["system_recall"],
                             "system_recall_top_occlusion_bucket":
                                 rep[name]["system_by_occlusion_bucket"][-1]["coverage"]}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# bundles
# ---------------------------------------------------------------------------
def basic_bundle(cfg: dict, t: Tables, class_names: list[str] | None, posthoc: PostHoc | None,
                 boot_w: np.ndarray, tau: float) -> dict:
    """Detection quality, the usability gate, calibration at the standard
    threshold, and (if a post-hoc layer is given) its conformal report.
    """
    ev = cfg["evaluation"]
    det = detection_summary(t, tau, class_names)
    usable, message = usable_for_calibration_analysis(det, float(ev.get("min_recall_at_floor", 0.5)))
    out = {"detection": det, "gate": {"usable": usable, "message": message},
           "calibration": operating_point(t, tau, int(ev.get("confidence_bins", 10)))}
    if posthoc is not None:
        out["conformal"] = conformal_report(t, posthoc.rule_inputs(t), tau,
                                            cfg["occlusion"]["bucket_edges"], boot_w,
                                            int(ev.get("min_bucket_n", 30)))
    return out


def train_class_counts(dataset: str, n_classes: int, processed_root: str | Path = "data/processed") -> np.ndarray:
    """Objects per class in the fit role (the rarity reference for RQ5)."""
    counts = np.zeros(n_classes)
    for _, label_path in role_images(dataset, "fit", processed_root):
        rows, _ = read_label_file(label_path, "bbox")
        for r in rows:
            if 0 <= r[0] < n_classes:
                counts[r[0]] += 1
    return counts
