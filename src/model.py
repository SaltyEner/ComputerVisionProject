"""Transfer-learning model factory.

A pretrained ImageNet backbone with its classification head swapped for one
sized to our number of classes. When the backbone is frozen only the head is
trained, which lets the training script use the fast "feature caching" path.

Supported backbones, lightest first:
    mobilenet_v3_small  (~2.5M params, default — best for a weak CPU)
    resnet18
    efficientnet_b0
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_BACKBONES = ("mobilenet_v3_small", "resnet18", "efficientnet_b0")


def build_model(
    num_classes: int,
    backbone: str = "mobilenet_v3_small",
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Create a backbone with a fresh classifier head."""
    if backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        in_features = net.classifier[0].in_features  # 576
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False
        # a single linear head keeps the feature-caching path simple
        net.classifier = nn.Linear(in_features, num_classes)

    elif backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        in_features = net.fc.in_features
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False
        net.fc = nn.Linear(in_features, num_classes)

    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False
        net.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown backbone {backbone!r}. Choose from {SUPPORTED_BACKBONES}."
        )

    return net


def get_classifier(model: nn.Module, backbone: str) -> nn.Module:
    """Return the trainable classifier head module."""
    if backbone == "mobilenet_v3_small":
        return model.classifier
    if backbone == "resnet18":
        return model.fc
    if backbone == "efficientnet_b0":
        return model.classifier[1]
    raise ValueError(f"Unknown backbone {backbone!r}.")


def set_classifier(model: nn.Module, backbone: str, head: nn.Module) -> None:
    """Replace the classifier head (used to swap in Identity for feature caching)."""
    if backbone == "mobilenet_v3_small":
        model.classifier = head
    elif backbone == "resnet18":
        model.fc = head
    elif backbone == "efficientnet_b0":
        model.classifier[1] = head
    else:
        raise ValueError(f"Unknown backbone {backbone!r}.")


def get_target_layer(model: nn.Module, backbone: str) -> nn.Module:
    """Return the last conv layer used as the Grad-CAM target."""
    if backbone == "mobilenet_v3_small":
        return model.features[-1]
    if backbone == "resnet18":
        return model.layer4[-1]
    if backbone == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unknown backbone {backbone!r}.")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
