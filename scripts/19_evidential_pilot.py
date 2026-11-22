#!/usr/bin/env python3
"""Phase 2 (v0.6): evidential YOLO26 pilot -- predictions with per-box evidence, and the analysis.

  python scripts/19_evidential_pilot.py --predict [--detector yolo26_evid_pilot]     (GPU, gpu_share)
      calib + test of the development dataset. Writes the usual record files
      outputs/<experiment>/<detector>/predictions/<dataset>__<role>.npz (scripts 06-08 can use them)
      and, aligned candidate by candidate, <dataset>__<role>__evidence.npz with alpha, beta of the
      box's predicted class (models/evidential_yolo.py).
  python scripts/19_evidential_pilot.py --analyse
      * AP50 (class-agnostic and class-aware) at the detector's F1-optimal threshold on calib, next to
        the yolo26 family (seeds 0-2) from their detection.json;
      * per vehicle (its matched candidate, class-agnostic matching), aleatoric and epistemic
        uncertainty, and their association with annotated occlusion o_j and with class rarity
        (-log of the class's share in the fit role), as Spearman and partial Spearman (controls:
        log relative box size), with image-bootstrap intervals;
      * the "uncertainty split": Spearman(aleatoric, epistemic) and the two differences
        rho(epistemic, rarity) - rho(aleatoric, rarity) and rho(aleatoric, o_j) - rho(epistemic, o_j).
      Writes outputs/<experiment>/<detector>/class_agnostic/results/evidential_pilot.json.
"""
from __future__ import annotations

import numpy as np
from _common import setup

from confdet.data.splits import processed_info, role_images
from confdet.evaluation.detection_metrics import detection_summary
from confdet.evaluation.records import load_records, save_records
from confdet.evaluation.stats import bootstrap_image_weights, partial_spearman, percentile_ci, resample_rows_by_image, spearman
from confdet.models.detector_factory import UltralyticsDetector, apply_memory_fraction, detector_config
from confdet.pipeline import collect_records, max_f1_threshold, tables_from_records, train_class_counts
from confdet.utils.io_utils import load_results, save_results
from confdet.utils.paths import detector_dir, predictions_path, processed_root, results_dir, weights_path


class EvidentialDetector(UltralyticsDetector):
    """Ultralytics prediction as usual, plus alpha/beta of every kept box (anchor indices from NMS)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        import ultralytics.utils.nms as nms_mod
        self._nms_mod, self._orig = nms_mod, nms_mod.non_max_suppression
        self._keep, self._evidence, self.extras = None, None, []

        def nms_wrapper(*args, **kwargs):
            want = kwargs.get("return_idxs", False)
            kwargs["return_idxs"] = True
            out, keep = self._orig(*args, **kwargs)
            self._keep = keep
            return (out, keep) if want else out

        nms_mod.non_max_suppression = nms_wrapper
        self._hooked = False

    def _hook(self):
        head = self.model.predictor.model.model.model[-1]

        def grab(_m, _i, out):
            preds = out[1] if isinstance(out, tuple) else out
            if isinstance(preds, dict) and "one2many" in preds:
                preds = preds["one2one"] if getattr(head, "end2end", False) and preds.get("one2one") else preds["one2many"]
            self._evidence = (preds["alpha"].detach().float().cpu().numpy(), preds["beta"].detach().float().cpu().numpy())

        head.register_forward_hook(grab)
        self._hooked = True

    def predict(self, source):
        if not self._hooked and getattr(self.model, "predictor", None) is not None:
            self._hook()
        xyxy, conf, cls = super().predict(source)
        if not self._hooked:                       # the first call builds the predictor: hook, then redo
            self._hook()
            xyxy, conf, cls = super().predict(source)
        a, b = self._evidence
        k = self._keep[0].cpu().numpy().astype(int) if len(conf) else np.zeros(0, int)
        self.extras.append((a[0, cls, k] if len(conf) else np.zeros(0), b[0, cls, k] if len(conf) else np.zeros(0)))
        return xyxy, conf, cls


def predict(cfg, det_name: str, log) -> None:
    dev, root = cfg["experiment"]["development"], processed_root(cfg)
    det = detector_config(cfg, det_name)
    share = cfg["detectors"].get("gpu_share") or {}
    apply_memory_fraction(share.get("memory_fraction"))
    p = cfg.get("prediction", {})
    d = EvidentialDetector(weights_path(cfg, det_name), det.family, det.imgsz, float(p.get("conf_floor", 0.001)),
                           float(p.get("nms_iou", 0.7)), int(p.get("max_det", 300)), duty_cycle=share.get("duty_cycle"))
    for role in ("calib", "test"):
        d.extras = []
        rs = collect_records(d, dev, role, cfg["occlusion"].get("proxy", "bottom_y"), root, log)
        rs.meta.update({"detector": det_name, "evidential": True})
        out = predictions_path(cfg, det_name, dev, role)
        save_records(out, rs)
        alpha = np.concatenate([e[0] for e in d.extras]) if d.extras else np.zeros(0)
        beta = np.concatenate([e[1] for e in d.extras]) if d.extras else np.zeros(0)
        assert len(alpha) == sum(len(r.cand_conf) for r in rs.records)
        tmp = out.with_name(out.stem + "__evidence.tmp.npz")
        np.savez_compressed(tmp, alpha=alpha.astype(np.float32), beta=beta.astype(np.float32))
        tmp.replace(out.with_name(out.stem + "__evidence.npz"))
        log.info(f"{det_name} {dev}/{role}: {len(rs)} images, {len(alpha)} candidates with evidence -> {out}")


def _ci_rho(fn, rows_img, n_images, n=200, seed=0):
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n):
        r = resample_rows_by_image(rows_img, n_images, rng)
        if len(r) > 10:
            reps.append(fn(r))
    return percentile_ci(reps)


def analyse(cfg, det_name: str, log) -> None:
    from confdet.models.evidential_yolo import beta_uncertainty
    dev, root = cfg["experiment"]["development"], processed_root(cfg)
    names = processed_info(dev, root).names
    rs_cal, rs_test = (load_records(predictions_path(cfg, det_name, dev, r)) for r in ("calib", "test"))
    ev = np.load(predictions_path(cfg, det_name, dev, "test").with_name(f"{dev}__test__evidence.npz"))
    tau, _ = max_f1_threshold(tables_from_records(cfg, rs_cal, "class_agnostic", context_conf=0.25))
    t = tables_from_records(cfg, rs_test, "class_agnostic", context_conf=tau)
    t_nat = tables_from_records(cfg, rs_test, "native", context_conf=tau)
    if len(ev["alpha"]) != t.n_cand:
        raise SystemExit("evidence is not aligned with the candidates")
    u = beta_uncertainty(ev["alpha"], ev["beta"])
    agn, nat = detection_summary(t, tau), detection_summary(t_nat, tau, names)
    family = {}
    for m in ("yolo26", "yolo26_s1", "yolo26_s2"):
        f = results_dir(cfg, m, "native") / "detection.json"
        if f.exists():
            d = load_results(f)
            family[m] = {"ap50_class_agnostic": d["class_agnostic"]["ap50"], "ap50_class_aware": d["class_aware"]["ap50"]}
    fam_agn = np.array([v["ap50_class_agnostic"] for v in family.values()])
    # per vehicle: its matched candidate's uncertainties (proposed vehicles only)
    prop = t.g_proposed
    cand = t.g_cand[prop]
    alea, epi = u["aleatoric"][cand], u["epistemic"][cand]
    occ = t.g_occ[prop]
    counts = train_class_counts(dev, len(names), root)
    freq = counts / max(counts.sum(), 1)
    rarity = -np.log(np.clip(freq[np.clip(t_nat.g_cls, 0, len(names) - 1)], 1e-9, None))[prop]
    size = t.g_area[prop][:, None]
    img = t.g_img[prop]
    res = {}
    for uname, uv in (("aleatoric", alea), ("epistemic", epi)):
        for vname, vv in (("annotated_occlusion", occ), ("class_rarity", rarity)):
            res[f"{uname}_vs_{vname}"] = {
                "spearman": spearman(uv, vv),
                "spearman_ci": _ci_rho(lambda r: spearman(uv[r], vv[r]), img, t.n_images),
                "partial_spearman_size": partial_spearman(uv, vv, size),
                "partial_ci": _ci_rho(lambda r: partial_spearman(uv[r], vv[r], size[r]), img, t.n_images)}
    split = {"spearman_aleatoric_vs_epistemic": spearman(alea, epi),
             "rarity_diff_epi_minus_alea": res["epistemic_vs_class_rarity"]["spearman"] - res["aleatoric_vs_class_rarity"]["spearman"],
             "rarity_diff_ci": _ci_rho(lambda r: spearman(epi[r], rarity[r]) - spearman(alea[r], rarity[r]), img, t.n_images),
             "occlusion_diff_alea_minus_epi": res["aleatoric_vs_annotated_occlusion"]["spearman"] - res["epistemic_vs_annotated_occlusion"]["spearman"],
             "occlusion_diff_ci": _ci_rho(lambda r: spearman(alea[r], occ[r]) - spearman(epi[r], occ[r]), img, t.n_images)}
    edges = cfg["occlusion"]["bucket_edges"]
    b = np.clip(np.searchsorted(edges, occ, side="right") - 1, 0, len(edges) - 2)
    by_bucket = [{"lo": edges[k], "hi": edges[k + 1], "n": int((b == k).sum()),
                  "mean_aleatoric": float(alea[b == k].mean()) if (b == k).any() else None,
                  "mean_epistemic": float(epi[b == k].mean()) if (b == k).any() else None} for k in range(len(edges) - 1)]
    out = {"detector": det_name, "threshold_max_f1_calib": tau,
           "ap50_class_agnostic": agn["ap50"], "ap50_class_aware": nat["ap50"],
           "recall_at_threshold": agn["recall_at_standard"], "fp_per_image": agn["fp_per_image_at_standard"],
           "yolo26_family": family, "yolo26_family_ap50_class_agnostic_mean": float(fam_agn.mean()) if len(fam_agn) else None,
           "ap50_gap_to_yolo26_family_mean": agn["ap50"] - float(fam_agn.mean()) if len(fam_agn) else None,
           "n_vehicles_with_candidate": int(prop.sum()), "associations": res, "uncertainty_split": split,
           "by_annotated_occlusion": by_bucket,
           "note": ("pilot trained with batch 4 / patience 10 under the GPU share; the yolo26 family used batch 8 / patience 25"
                    if det_name.startswith("yolo26_evid_pilot") else
                    "final v0.6 model, trained on Beocat (L40S) with the same batch and patience as the yolo26 family "
                    "(configs/beocat_batch.yaml; patience 10)")}
    save_results(out, results_dir(cfg, det_name, "class_agnostic") / "evidential_pilot.json")
    log.info(f"{det_name}: AP50 agnostic {agn['ap50']:.4f} (yolo26 family {out['yolo26_family_ap50_class_agnostic_mean']}) | "
             f"aware {nat['ap50']:.4f} | rho(alea, epi) {split['spearman_aleatoric_vs_epistemic']:.3f} | "
             + " | ".join(f"{k} {v['spearman']:.3f}" for k, v in res.items()))


def main():
    def extra(p):
        p.add_argument("--predict", action="store_true")
        p.add_argument("--analyse", action="store_true")
        p.add_argument("--detector", default="yolo26_evid_pilot")

    args, cfg, log = setup("evidential_pilot", __doc__, extra=extra)
    if args.predict:
        predict(cfg, args.detector, log)
    if args.analyse:
        analyse(cfg, args.detector, log)


if __name__ == "__main__":
    main()
