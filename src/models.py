"""Modele z wagami ImageNet i różną głębokością fine-tuningu."""
import torch.nn as nn
from torchvision import models

import config as C


def build_model(arch: str, finetune: str, num_classes: int = C.NUM_CLASSES, pretrained: bool = True):
    """Zwraca (model, head_params, body_params, frozen_modules).

    finetune: 'frozen' — tylko nowa głowa; 'last_block' — ostatni blok + głowa; 'full' — całość.
    frozen_modules to moduły trzymane w trybie eval (statystyki BatchNorm nie są aktualizowane).
    """
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
        blocks = [m.conv1, m.bn1, m.layer1, m.layer2, m.layer3, m.layer4]
    elif arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        head = m.classifier
        # ostatni "blok" = dwa ostatnie inverted residual + konwolucja 1x1 (features[16:])
        blocks = [m.features[:16], m.features[16:]]
    else:
        raise ValueError(arch)

    n_open = {"frozen": 0, "last_block": 1, "full": len(blocks)}[finetune]
    frozen, opened = blocks[: len(blocks) - n_open], blocks[len(blocks) - n_open:]
    for mod in frozen:
        for p in mod.parameters():
            p.requires_grad = False
    head_params = list(head.parameters())
    body_params = [p for mod in opened for p in mod.parameters()]
    return m, head_params, body_params, frozen


def count_params(model) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
