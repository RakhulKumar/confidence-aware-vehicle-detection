"""Evidential (Beta) head for YOLO26 (models/evidential_yolo.py): the maths, and that the converted
Ultralytics model trains a step, keeps its outputs consistent, survives save/load and resume-load.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.models import evidential_yolo as ev  # noqa: E402


def test_beta_uncertainty_decomposition():
    u = ev.beta_uncertainty([1, 10, 100, 2], [1, 10, 100, 50])
    assert u["p"][:3] == pytest.approx([0.5, 0.5, 0.5])
    assert np.allclose(u["aleatoric"] + u["epistemic"], u["total"])
    assert u["epistemic"][0] > u["epistemic"][1] > u["epistemic"][2]          # more evidence, less epistemic
    assert u["aleatoric"][2] > u["aleatoric"][0]                               # 50/50 with lots of evidence: aleatoric
    assert u["evidence"].tolist() == [0, 18, 198, 50]


def test_evidential_loss_rewards_correct_evidence_and_kl_targets_misleading_evidence():
    import torch
    a, b = torch.tensor([20.0, 1.0, 1.0]), torch.tensor([1.0, 20.0, 1.0])
    pos = ev.evidential_beta_loss(a, b, torch.ones(3), 0.0)
    assert pos[0] < pos[2] < pos[1]                                            # right > uncertain > wrong
    neg = ev.evidential_beta_loss(a, b, torch.zeros(3), 0.0)
    assert neg[1] < neg[2] < neg[0]
    # KL is zero for the uniform Beta and penalises evidence for the wrong outcome only
    assert ev.evidential_beta_loss(torch.ones(1), torch.ones(1), torch.zeros(1), 1.0).item() == pytest.approx(
        ev.evidential_beta_loss(torch.ones(1), torch.ones(1), torch.zeros(1), 0.0).item())
    k_wrong = (ev.evidential_beta_loss(a[:1], b[:1], torch.zeros(1), 1.0) - ev.evidential_beta_loss(a[:1], b[:1], torch.zeros(1), 0.0)).item()
    k_right = (ev.evidential_beta_loss(a[1:2], b[1:2], torch.zeros(1), 1.0) - ev.evidential_beta_loss(a[1:2], b[1:2], torch.zeros(1), 0.0)).item()
    assert k_wrong > 0.5 and abs(k_right) < 1e-6
    assert ev.kl_weight(0) == 0 and ev.kl_weight(5) == pytest.approx(ev.KL_MAX / 2) and ev.kl_weight(50) == ev.KL_MAX


def _tiny_model(nc=3):
    from ultralytics.nn.tasks import DetectionModel
    return DetectionModel("yolo26n.yaml", nc=nc, verbose=False)


def test_converted_model_outputs_expected_probability_and_trains_a_step(tmp_path):
    import torch
    from types import SimpleNamespace
    m = ev.convert_to_evidential(_tiny_model())
    c = ev.classes()
    assert isinstance(m, c["model"]) and isinstance(m.model[-1], c["detect"])
    m.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=10)
    x = torch.zeros(2, 3, 128, 128)
    m.train()
    preds = m(x)
    o2m = preds["one2many"]
    assert o2m["alpha"].shape == o2m["scores"].shape and (o2m["alpha"] >= 1).all()
    assert torch.allclose(o2m["scores"].sigmoid(), o2m["alpha"] / (o2m["alpha"] + o2m["beta"]), atol=1e-4)
    assert (o2m["scores"].sigmoid() < 0.5).all()                     # starts with little evidence for a vehicle
    batch = {"batch_idx": torch.tensor([0.0, 1.0]), "cls": torch.tensor([[1.0], [2.0]]),
             "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.4, 0.6, 0.2, 0.2]]), "img": x}
    loss, items = m.loss(batch, preds)
    assert torch.isfinite(loss).all() and loss.sum() > 0
    loss.sum().backward()
    head = m.model[-1]
    assert head.cv3[0][-1].weight.grad is not None and head.cv3[0][-1].weight.shape[0] == 6
    # inference keeps the usual (boxes + nc scores) layout, so validation, prediction and NMS work unchanged
    m.eval()
    y, _ = m(x)
    assert y.shape[1] == 4 + 3


def test_checkpoint_round_trip_and_resume_loading(tmp_path):
    import torch
    m = ev.convert_to_evidential(_tiny_model())
    path = tmp_path / "w.pt"
    torch.save({"model": m}, path)
    loaded = torch.load(path, weights_only=False)["model"]
    assert type(loaded).__name__ == "EvidentialDetectionModel"
    # what the trainer does on resume: build, convert, load -> every head weight survives
    fresh = ev.convert_to_evidential(_tiny_model())
    fresh.load(loaded)
    for a, b in zip(fresh.model[-1].cv3[0][-1].parameters(), m.model[-1].cv3[0][-1].parameters()):
        assert torch.equal(a, b)


def test_initial_probability_matches_the_ultralytics_prior():
    """Every anchor starts at p ~ 0.01, as Ultralytics' bias_init does for its BCE head
    (the first pilot used p ~ 0.15, which floods early training with confident background)."""
    import torch
    m = ev.convert_to_evidential(_tiny_model())
    m.train()
    o2m = m(torch.zeros(1, 3, 128, 128))["one2many"]
    p = o2m["scores"].sigmoid()
    assert float(p.max()) < 0.02 and float(p.min()) > 0.005
