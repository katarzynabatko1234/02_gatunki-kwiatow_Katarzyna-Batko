import torch
import torch.nn as nn

from models import build_model
from train import set_train_mode


def _steps(finetune, n=5):
    torch.manual_seed(6102)
    model, head, body, frozen = build_model("resnet18", finetune, num_classes=5, pretrained=False)
    params = [{"params": head, "lr": 1e-2}] + ([{"params": body, "lr": 1e-3}] if body else [])
    opt = torch.optim.AdamW(params)
    x, y = torch.randn(8, 3, 64, 64), torch.randint(0, 5, (8,))
    frozen_before = [p.clone() for m in frozen for p in m.parameters()]
    bn_before = [b.clone() for m in frozen for b in m.buffers()]
    losses = []
    for _ in range(n):
        set_train_mode(model, frozen)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    frozen_after = [p for m in frozen for p in m.parameters()]
    bn_after = [b for m in frozen for b in m.buffers()]
    return losses, frozen_before, frozen_after, bn_before, bn_after, model, frozen


def test_frozen_training_overfits_batch_and_keeps_body():
    losses, fb, fa, bb, ba, model, frozen = _steps("frozen", n=10)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]
    assert all(torch.equal(a, b) for a, b in zip(fb, fa))  # wagi ciała bez zmian
    assert all(torch.equal(a, b) for a, b in zip(bb, ba))  # statystyki BN ciała bez zmian
    assert all(not m.training for m in frozen) and model.fc.training


def test_last_block_updates_only_layer4():
    losses, fb, fa, *_ = _steps("last_block")
    assert losses[-1] < losses[0]
    assert all(torch.equal(a, b) for a, b in zip(fb, fa))
