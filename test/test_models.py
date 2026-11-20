import pytest
import torch

from models import build_model, count_params

EXPECTED_TRAINABLE = {
    ("resnet18", "frozen"): 52_326,
    ("resnet18", "last_block"): 8_446_054,
    ("resnet18", "full"): 11_228_838,
    ("mobilenet_v2", "frozen"): 130_662,
    ("mobilenet_v2", "full"): 2_354_534,
}


@pytest.mark.parametrize("arch,finetune", list(EXPECTED_TRAINABLE))
def test_trainable_params(arch, finetune):
    model, head, body, frozen = build_model(arch, finetune, pretrained=False)
    total, trainable = count_params(model)
    assert trainable == EXPECTED_TRAINABLE[(arch, finetune)]
    assert sum(p.numel() for p in head + body) == trainable
    assert all(not p.requires_grad for m in frozen for p in m.parameters())


@pytest.mark.parametrize("arch", ["resnet18", "mobilenet_v2"])
def test_output_shape(arch):
    model, *_ = build_model(arch, "frozen", num_classes=102, pretrained=False)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 102)
