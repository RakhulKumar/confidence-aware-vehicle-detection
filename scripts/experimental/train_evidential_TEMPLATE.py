#!/usr/bin/env python3
"""NOT PART OF THE VERIFIED PIPELINE.  Unfinished template, kept for reference.

Status, stated plainly:
  * src/confdet/uq/evidential.py (Dirichlet head + loss) has never been run
    against real features; it needs PyTorch and was not testable where the
    rest of this codebase was verified.
  * this script builds the head and stops: the feature-extraction loop for a
    specific detector is not written.
  * nothing in scripts/02..11 depends on it.

RQ5 no longer waits for this: scripts/07 answers it from the frozen detector's
raw output (src/confdet/evaluation/rq5_miss_causes.py). An evidential head
remains a possible ablation for the journal version; treat everything below
as a starting sketch, not as working code.

--- original docstring ---
Stage 3 (ablation variant): train the evidential-learning classification
head as a comparison against the primary conformal pipeline
(scripts/05_calibrate_conformal.py). Unlike conformal prediction, this
REQUIRES retraining — see docs/theoretical_foundations.md §3 for why it's
kept as a secondary/ablation mechanism rather than the primary one.

This script trains ONLY the evidential classification head described in
src/confdet/uq/evidential.py, attached to features extracted from the
already-fine-tuned Stage 1 detector (frozen backbone) — not a full detector
retrain. This keeps the ablation's compute cost proportionate to what it's
actually testing (the UQ head), matching the plan's Section 7 risk
mitigation on compute budget.

Requires: PyTorch (pip install torch, see requirements.txt for the CUDA 13
wheel index) and a completed scripts/04_train_detector.py run to extract
features from.

Usage:
  python scripts/06_train_evidential.py --weights outputs/checkpoints/yolo26_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from confdet.utils.io_utils import load_config  # noqa: E402
from confdet.utils.logging_utils import get_logger  # noqa: E402
from confdet.utils.seed import set_seed  # noqa: E402

logger = get_logger("train_evidential")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--weights", required=True, help="Fine-tuned Stage 1 detector weights")
    parser.add_argument("--num-classes", type=int, required=True,
                         help="Number of vehicle classes (nc from the primary dataset's data.yaml)")
    parser.add_argument("--feature-dim", type=int, default=256,
                         help="Backbone feature dimension the evidential head attaches to "
                              "(check the specific detector's neck output — this varies by "
                              "architecture/scale and is NOT auto-detected here; verify "
                              "against the actual checkpoint before trusting the default)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    ev_cfg = cfg["evidential"]

    try:
        import torch
    except ImportError:
        logger.error(
            "PyTorch is required for this script. Install it first: see "
            "requirements.txt / data/README.md for the CUDA 13 wheel index, "
            "e.g. pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu130"
        )
        sys.exit(1)

    from confdet.uq.evidential import EvidentialHead, evidential_loss, evidential_output  # noqa: E402

    logger.warning(
        "This script is a TEMPLATE for the evidential training loop, not a "
        "complete one: feature extraction from a specific detector checkpoint "
        "(YOLO26/YOLOv8/RT-DETR all expose backbone features differently) is "
        "architecture-specific glue that needs to be written against the "
        "actual checkpoint format once Stage 1 training has produced real "
        "weights (see the 'Open Questions' in the original research plan re: "
        "base detector choice -- this script's feature-extraction hook "
        "depends on that choice being finalized). The loss function and "
        "evidential head themselves (src/confdet/uq/evidential.py) are complete "
        "and unit-testable independent of that glue."
    )

    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
    head = EvidentialHead.build(in_features=args.feature_dim, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=ev_cfg["lr"])

    logger.info(
        f"Evidential head built on {device}: {args.feature_dim} -> {args.num_classes} classes. "
        f"Training for {ev_cfg['epochs']} epochs once a feature-extraction DataLoader is "
        f"wired in above (see warning)."
    )
    # --- feature-extraction DataLoader + training loop goes here, once the
    # chosen detector's feature-extraction hook is written (see warning above).
    # The per-batch call is:
    #   evidence = head(features)                                  # (N, K)
    #   loss = evidential_loss(evidence, target_onehot, epoch,
    #                          ev_cfg['lambda_final'], ev_cfg['lambda_anneal_epochs'])
    #   loss.backward(); optimizer.step(); optimizer.zero_grad()


if __name__ == "__main__":
    main()
