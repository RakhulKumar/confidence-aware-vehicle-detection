#!/usr/bin/env python3
"""Controlled occlusion probe (a diagnostic; see data/synthetic_occlusion.py).

The same vehicles are shown to the detector at occlusion 0, 0.1, ..., 0.9 by
pasting a real vehicle crop over the bottom of each one. Everything except
occlusion is held fixed, so the rise in misses is caused by the occlusion.

  python scripts/09_synthetic_occlusion_probe.py [--detector NAME] [--dataset NAME]
         [--role test] [--n-targets 150]

Needs a trained Ultralytics detector (images are built in memory). Target
vehicles come from --dataset (default: the development dataset), from a role
the detector did not train on. Writes
outputs/<experiment>/<detector>/results/synthetic_probe_<dataset>.json and a
figure, both labelled as a controlled probe -- never mix them into the real
evaluation numbers.
"""
from __future__ import annotations

from _common import setup

from confdet.data.splits import processed_info, role_images
from confdet.data.synthetic_occlusion import analyse_probe, probe_predictions, score_probe, select_targets
from confdet.models.detector_factory import detector_config, load_detector
from confdet.pipeline import PostHoc
from confdet.utils.io_utils import load_results, matching_modes, save_results
from confdet.utils.paths import calibration_path, figures_dir, processed_root, results_dir, weights_path
from confdet.visualization import plots


def main():
    def extra(p):
        p.add_argument("--dataset", default=None)
        p.add_argument("--role", default="test", choices=["val", "calib", "test"])
        p.add_argument("--n-targets", type=int, default=150)
        p.add_argument("--matching", default=None, help="one matching mode (default: every mode)")
        p.add_argument("--severities", nargs="+", type=float,
                       default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    args, cfg, log = setup("synthetic_probe", __doc__, detector=True, extra=extra)
    dev = cfg["experiment"]["development"]
    dataset = args.dataset or dev
    if dataset != dev:
        raise SystemExit("probe targets must come from the development dataset (the one the detector "
                         "was trained on)")
    root = processed_root(cfg)
    det = detector_config(cfg, args.detector)
    if det.family == "simulated":
        raise SystemExit("the probe needs a real detector that looks at pixels")
    detector = load_detector(det, weights_path(cfg, det.name), cfg.get("prediction", {}),
                             processed_info(dev, root).names, gpu_share=cfg["detectors"].get("gpu_share"))
    if args.severities[0] != 0.0:
        raise SystemExit("--severities must start at 0.0 (the unoccluded reference)")
    seed = int(cfg["project"].get("seed", 42))
    targets = select_targets(role_images(dataset, args.role, root), args.n_targets, seed=seed)
    log.info(f"{len(targets)} isolated target vehicles from {dataset}/{args.role}; "
             f"{len(targets) * len(args.severities)} probe images ...")
    ev = cfg["evaluation"]
    preds = probe_predictions(detector.predict, targets, args.severities)      # one detector pass, all modes
    for mode in ([args.matching] if args.matching else matching_modes(cfg)):
        tau = PostHoc(load_results(calibration_path(cfg, det.name, mode))).tau
        scores = score_probe(preds, targets, float(ev.get("iou_match_threshold", 0.5)),
                             class_aware=(mode != "class_agnostic"))
        result = analyse_probe(scores, args.severities, tau, seed=seed)
        result["matching"] = mode
        result["scoring_rule"] = ("a candidate counts for the target if IoU(target) >= iou_match_threshold "
                                  "and IoU(target) > IoU(pasted occluder box)")
        result["note_scope"] = ("CONTROLLED PROBE with pasted occluders -- constructed severity, not real "
                                "held-out traffic. Report separately from RQ1-RQ5.")
        result.update({"dataset": dataset, "role": args.role, "detector": det.name})
        out = results_dir(cfg, det.name, mode) / f"synthetic_probe_{dataset}.json"
        save_results(result, out)
        if result["by_severity"]:
            plots.save(plots.plot_synthetic_probe(result),
                       figures_dir(cfg, det.name, mode) / f"synthetic_probe_{dataset}.png")
            log.info(f"[{mode}, tau {tau:.2f}] miss rate by constructed severity: " + "  ".join(
                f"{b['severity']:.1f}:{b['miss_rate']:.2f}" for b in result["by_severity"]))
            log.info(f"[{mode}] within-vehicle slope of uncertainty on severity: "
                     f"{result['mean_within_target_slope']:.3f} CI {result['slope_ci']}")
        else:
            log.warning(result.get("note"))
        log.info(f"Saved {out}")


if __name__ == "__main__":
    main()
