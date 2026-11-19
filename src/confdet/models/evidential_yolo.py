"""Evidential (Beta) classification head for Ultralytics YOLO26 -- v0.6 Phase 2 pilot.

Per anchor and class the head outputs TWO evidence values instead of one logit:
    alpha = 1 + softplus(a),  beta = 1 + softplus(b),  S = alpha + beta
    p     = alpha / S                         (expected probability; what NMS, assignment and
                                               validation see, as logit(p) in place of the logit)
    aleatoric = E[p (1 - p)]  = alpha beta / (S (S + 1))       (expected Bernoulli variance)
    epistemic = Var[p]        = alpha beta / (S^2 (S + 1))     (spread of p itself)
    aleatoric + epistemic = p (1 - p)
Loss (Sensoy et al. 2018, binary/Beta form) per anchor and class, soft target y in [0, 1]:
    y (psi(S) - psi(alpha)) + (1 - y) (psi(S) - psi(beta))            (expected cross-entropy)
  + lambda_t KL( Beta(1 + (1-y)(alpha-1), 1 + y(beta-1)) || Beta(1, 1) )   (misleading evidence)
    lambda_t = kl_max * min(1, epoch / anneal_epochs)    [kl_max 0.1, anneal 10 epochs; fixed in advance]
It replaces the classification BCE inside Ultralytics' v8DetectionLoss (both the one-to-many and
the one-to-one branch of YOLO26's E2ELoss); box and DFL losses are unchanged. Initialisation puts
the beta evidence high (p ~ 0.01) so the head starts like a normal detector (p small everywhere).

Pure numpy helpers (beta_uncertainty) are importable without torch; everything else needs torch
and ultralytics and is imported lazily.
"""
from __future__ import annotations

import numpy as np

KL_MAX = 0.1
ANNEAL_EPOCHS = 10
BETA_INIT = 98.0       # softplus(98) = 98 -> beta ~ 99, alpha ~ 1 -> p ~ 0.01 at the first step: the same prior
                       # Ultralytics gives its BCE head (bias_init). The first v0.6 pilot ran with 4.6 (p ~ 0.15),
                       # contradicting this module's own specification; corrected 18 Nov before any test result existed.
STATE = {"epoch": 0}   # set by a training callback; read by the loss (KL annealing)


def beta_uncertainty(alpha, beta) -> dict:
    a, b = np.asarray(alpha, float), np.asarray(beta, float)
    s = a + b
    p = a / s
    alea = a * b / (s * (s + 1))
    epi = a * b / (s ** 2 * (s + 1))
    return {"p": p, "aleatoric": alea, "epistemic": epi, "total": p * (1 - p), "evidence": s - 2}


def kl_weight(epoch: int) -> float:
    return KL_MAX * min(1.0, float(epoch) / ANNEAL_EPOCHS)


def evidential_beta_loss(alpha, beta, y, lam: float):
    """Element-wise loss (torch tensors, same shape)."""
    import torch
    alpha, beta, y = alpha.float(), beta.float(), y.float()
    s = alpha + beta
    ece = y * (torch.digamma(s) - torch.digamma(alpha)) + (1 - y) * (torch.digamma(s) - torch.digamma(beta))
    if lam <= 0:
        return ece
    a = 1 + (1 - y) * (alpha - 1)
    b = 1 + y * (beta - 1)
    ln_b = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    kl = -ln_b + (a - 1) * torch.digamma(a) + (b - 1) * torch.digamma(b) - (a + b - 2) * torch.digamma(a + b)
    return ece + lam * kl


# ---------------------------------------------------------------------------
# Ultralytics glue (built lazily so importing this module never needs torch)
# ---------------------------------------------------------------------------
def _classes():
    import torch
    import torch.nn.functional as F
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
    from ultralytics.utils.tal import make_anchors

    class EvidentialDetect(Detect):
        def forward_head(self, x, box_head=None, cls_head=None):
            if box_head is None or cls_head is None:
                return {}
            bs = x[0].shape[0]
            boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
            raw = torch.cat([cls_head[i](x[i]).view(bs, 2 * self.nc, -1) for i in range(self.nl)], dim=-1)
            alpha = 1 + F.softplus(raw[:, : self.nc].float())
            beta = 1 + F.softplus(raw[:, self.nc:].float())
            p = (alpha / (alpha + beta)).clamp(1e-6, 1 - 1e-6)
            scores = torch.log(p / (1 - p)).to(raw.dtype)          # logit(p): sigmoid(scores) == p downstream
            return {"boxes": boxes, "scores": scores, "alpha": alpha, "beta": beta, "feats": x}

    class EvidentialDetectionLoss(v8DetectionLoss):
        def get_assigned_targets_and_loss(self, preds, batch):
            loss = torch.zeros(3, device=self.device)
            pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
            pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
            alpha = preds["alpha"].permute(0, 2, 1).contiguous()
            beta = preds["beta"].permute(0, 2, 1).contiguous()
            anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
            dtype = pred_scores.dtype
            batch_size = pred_scores.shape[0]
            imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
            pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
            _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
            target_scores_sum = target_scores.sum().clamp_(min=1)
            cls = evidential_beta_loss(alpha, beta, target_scores, kl_weight(STATE["epoch"]))
            if self.class_weights is not None:
                cls = cls * self.class_weights
            loss[1] = cls.sum() / target_scores_sum
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor)
            loss[0] *= self.hyp.box
            loss[1] *= self.hyp.cls
            loss[2] *= self.hyp.dfl
            return ((fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss,
                    dict(zip(self.loss_names, loss.detach())))

    class EvidentialDetectionModel(DetectionModel):
        def init_criterion(self):
            if getattr(self.model[-1], "one2one_cv2", None) is not None:
                return E2ELoss(self, EvidentialDetectionLoss)
            return EvidentialDetectionLoss(self)

    return EvidentialDetect, EvidentialDetectionLoss, EvidentialDetectionModel


_CACHE: dict = {}


def classes():
    if not _CACHE:
        d, l, m = _classes()
        # register under this module so checkpoints pickle/unpickle as confdet.models.evidential_yolo.<Name>
        for c in (d, l, m):
            c.__module__ = __name__
            c.__qualname__ = c.__name__
            globals()[c.__name__] = c
        _CACHE.update(detect=d, loss=l, model=m)
    return _CACHE


def __getattr__(name):            # lets torch.load find the classes when it unpickles a checkpoint
    if name in ("EvidentialDetect", "EvidentialDetectionLoss", "EvidentialDetectionModel"):
        return classes()[{"EvidentialDetect": "detect", "EvidentialDetectionLoss": "loss",
                          "EvidentialDetectionModel": "model"}[name]]
    raise AttributeError(name)


def convert_to_evidential(model):
    """Turn a built Ultralytics DetectionModel into the evidential variant, in place."""
    import torch.nn as nn
    c = classes()
    head = model.model[-1]
    nc = head.nc
    branches = [head.cv3] + ([head.one2one_cv3] if getattr(head, "one2one_cv3", None) is not None else [])
    for branch in branches:
        for seq in branch:
            old = seq[-1]
            new = nn.Conv2d(old.in_channels, 2 * nc, 1).to(old.weight.device, old.weight.dtype)
            with __import__("torch").no_grad():
                new.weight[:nc].copy_(old.weight)
                new.bias[:nc].copy_(old.bias)
                new.weight[nc:].zero_()
                new.bias[nc:].fill_(BETA_INIT)
            seq[-1] = new
    head.__class__ = c["detect"]
    model.__class__ = c["model"]
    return model


def evidential_trainer():
    """A DetectionTrainer whose model is converted to the evidential head."""
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import RANK

    class EvidentialTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            # convert BEFORE loading: on resume the checkpoint already has the 2*nc evidence head, and
            # loading it into a standard head would silently drop those weights (shape mismatch)
            model = self.set_model_names_for_load(
                DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1))
            model = convert_to_evidential(model)
            if weights:
                model.load(weights)
            return model

    return EvidentialTrainer


def epoch_callback(trainer):
    STATE["epoch"] = int(getattr(trainer, "epoch", 0))
