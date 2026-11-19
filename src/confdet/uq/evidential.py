"""Evidential deep learning (Dirichlet evidence head and loss).

STATUS: NOT PART OF THE VERIFIED PIPELINE. This file needs PyTorch and has
never been run against real detector features; no script in scripts/02..11
imports it. It is kept as the starting point for an optional ablation
(scripts/experimental/train_evidential_TEMPLATE.py). RQ5 is answered without
it, from the frozen detector's raw output (evaluation/rq5_miss_causes.py).

Retained specifically as an ablation baseline against the primary conformal
pipeline (src/confdet/uq/conformal.py): unlike conformal prediction, this
requires retraining the classification head with a modified loss, but in
exchange gives a direct, per-prediction aleatoric/epistemic decomposition,
which is the second, independent handle this plan uses for RQ5 (separating
occlusion-driven misses from rarity-driven misses).

Requires torch. Import is deferred so the rest of the package (occlusion,
calibration metrics, conformal) stays importable without a torch install —
useful for quick CI checks and for anyone who only wants the CP path.
"""
from __future__ import annotations

from dataclasses import dataclass


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except ImportError as e:  # pragma: no cover - exercised only without torch installed
        raise ImportError(
            "src/confdet/uq/evidential.py requires PyTorch. Install it first "
            "(see requirements.txt / data/README.md), or use the conformal "
            "prediction path (src/confdet/uq/conformal.py) which has no "
            "torch dependency for the calibration math itself."
        ) from e


@dataclass(frozen=True)
class EvidentialOutput:
    alpha: "object"        # torch.Tensor, shape (..., K) — Dirichlet parameters
    prob: "object"         # torch.Tensor, shape (..., K) — expected class probabilities
    aleatoric: "object"    # torch.Tensor, shape (...,) — expected Dirichlet variance
    epistemic: "object"    # torch.Tensor, shape (...,) — vacuity K/S


def evidence_to_dirichlet(evidence):
    """evidence -> alpha = evidence + 1, following Sensoy et al. (2018).
    `evidence` must already be non-negative (apply softplus/ReLU beforehand
    in the model head — this function does not clip).
    """
    torch, nn, F = _require_torch()
    return evidence + 1.0


def evidential_output(evidence) -> EvidentialOutput:
    """Full forward-pass decomposition from raw non-negative evidence.
    docs/theoretical_foundations.md §3, aleatoric/epistemic formulas.
    """
    torch, nn, F = _require_torch()
    alpha = evidence_to_dirichlet(evidence)
    S = alpha.sum(dim=-1, keepdim=True)
    prob = alpha / S
    K = alpha.shape[-1]
    aleatoric = (prob * (1 - prob)).sum(dim=-1)
    epistemic = (K / S).squeeze(-1)
    return EvidentialOutput(alpha=alpha, prob=prob, aleatoric=aleatoric, epistemic=epistemic)


def evidential_loss(evidence, target_onehot, epoch: int, lambda_final: float = 1.0,
                     lambda_anneal_epochs: int = 20):
    """Type-II maximum likelihood (Bayes risk) + annealed KL evidence
    regularizer. docs/theoretical_foundations.md §3.

    Args:
      evidence: (N, K) non-negative evidence from the model head.
      target_onehot: (N, K) one-hot ground-truth labels.
      epoch: current training epoch, for the KL annealing schedule.
      lambda_final: KL term weight once fully annealed in.
      lambda_anneal_epochs: epochs over which the KL weight ramps 0 -> lambda_final.
    """
    torch, nn, F = _require_torch()
    alpha = evidence_to_dirichlet(evidence)
    S = alpha.sum(dim=-1, keepdim=True)
    prob = alpha / S

    # Bayes-risk term: sum_k (y_k - p_k)^2 + p_k(1-p_k)/(S+1)
    bayes_risk = ((target_onehot - prob) ** 2 + prob * (1 - prob) / (S + 1)).sum(dim=-1)

    # KL(Dir(alpha_tilde) || Dir(1,...,1)), alpha_tilde zeroes evidence on the true class
    alpha_tilde = target_onehot + (1 - target_onehot) * alpha
    K = alpha.shape[-1]
    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
    kl = (
        torch.lgamma(S_tilde.squeeze(-1))
        - torch.lgamma(torch.tensor(float(K), device=alpha.device))
        - torch.lgamma(alpha_tilde).sum(dim=-1)
        + ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=-1)
    )

    lambda_t = min(lambda_final, lambda_final * epoch / max(1, lambda_anneal_epochs))
    return (bayes_risk + lambda_t * kl).mean()


class EvidentialHead:
    """Thin wrapper factory — not a full nn.Module subclass here to avoid a
    hard torch import at module load time (see _require_torch above). Call
    `build(in_features, num_classes)` to get an actual nn.Module once torch
    is confirmed available; wire its output into `evidential_output` /
    `evidential_loss` in the training loop (scripts/06_train_evidential.py).
    """

    @staticmethod
    def build(in_features: int, num_classes: int):
        torch, nn, F = _require_torch()

        class _Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(in_features, num_classes)

            def forward(self, x):
                # softplus keeps evidence non-negative and smooth (vs. ReLU's
                # hard zero, which can stall gradients for dead units early
                # in evidential training).
                return F.softplus(self.linear(x))

        return _Head()
