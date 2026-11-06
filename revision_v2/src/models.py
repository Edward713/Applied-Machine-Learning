"""Model factory, layer-wise parameter groups and freezing utilities.

A single ``build_model`` is used for training, testing and Grad-CAM, so the
architecture is never redefined (the previous notebook duplicated it).
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

# Parameter-name prefixes treated as "early" backbone (lower learning rate).
_EARLY = {
    "densenet121": ("features.conv0", "features.norm0", "features.denseblock1",
                    "features.transition1", "features.denseblock2", "features.transition2"),
    "resnet50": ("conv1", "bn1", "layer1", "layer2"),
    "resnet18": ("conv1", "bn1", "layer1", "layer2"),
    "simple_cnn": ("features.0", "features.1"),
}
_HEAD = {"densenet121": "classifier", "resnet50": "fc", "resnet18": "fc", "simple_cnn": "classifier"}


class SimpleCNN(nn.Module):
    """Small from-scratch baseline: 4 conv blocks + global average pooling."""

    def __init__(self, dropout: float = 0.4, width: int = 32):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, width), block(width, 2 * width),
                                      block(2 * width, 4 * width), block(4 * width, 8 * width))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(8 * width, 1))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


def build_model(cfg: dict) -> nn.Module:
    name, pretrained, p = cfg["model"]["name"], cfg["model"]["pretrained"], cfg["model"]["dropout"]
    if name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(p), nn.Linear(m.classifier.in_features, 1))
    elif name in ("resnet50", "resnet18"):
        ctor = getattr(models, name)
        weights = (models.ResNet50_Weights.DEFAULT if name == "resnet50"
                   else models.ResNet18_Weights.DEFAULT) if pretrained else None
        m = ctor(weights=weights)
        m.fc = nn.Sequential(nn.Dropout(p), nn.Linear(m.fc.in_features, 1))
    elif name == "simple_cnn":
        m = SimpleCNN(dropout=p)
    else:
        raise ValueError(f"Unknown model '{name}'")
    return m


def is_head(model_name: str, param_name: str) -> bool:
    return param_name.startswith(_HEAD[model_name])


def param_groups(model: nn.Module, cfg: dict) -> list[dict]:
    """Three AdamW groups: early backbone, late backbone, classification head."""
    name, o = cfg["model"]["name"], cfg["optim"]
    early, late, head = [], [], []
    for pname, param in model.named_parameters():
        if is_head(name, pname):
            head.append(param)
        elif pname.startswith(_EARLY[name]):
            early.append(param)
        else:
            late.append(param)
    return [{"params": early, "lr": o["lr_early"], "name": "early"},
            {"params": late, "lr": o["lr_late"], "name": "late"},
            {"params": head, "lr": o["lr_head"], "name": "head"}]


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    for pname, param in model.named_parameters():
        if not is_head(model_name, pname):
            param.requires_grad = trainable


def gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Last convolutional feature block, used by Grad-CAM."""
    if model_name == "densenet121":
        return model.features
    if model_name.startswith("resnet"):
        return model.layer4
    return model.features
