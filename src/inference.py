"""Shared inference helpers used by both the Streamlit app and the REST API.

Loads a trained checkpoint, runs prediction on a PIL image and produces a
Grad-CAM overlay. Keeping this in one place avoids duplicating logic between
the two front-ends.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import config
from .data import build_transforms
from .gradcam import GradCAM
from .model import build_model, get_device, get_target_layer


@dataclass
class LoadedModel:
    model: torch.nn.Module
    classes: list[str]
    backbone: str
    img_size: int
    device: torch.device


def load_model(checkpoint_path: str | Path = config.CHECKPOINT_PATH) -> LoadedModel:
    """Rebuild the network from a checkpoint saved by train.py."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train the model first: "
            "`python -m src.train`"
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    classes: list[str] = ckpt["classes"]
    backbone: str = ckpt["backbone"]
    img_size: int = ckpt["img_size"]

    model = build_model(
        num_classes=len(classes),
        backbone=backbone,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    device = get_device()
    model.to(device).eval()
    return LoadedModel(model, classes, backbone, img_size, device)


def _preprocess(img: Image.Image, img_size: int) -> torch.Tensor:
    tfm = build_transforms(img_size, train=False)
    return tfm(img.convert("RGB")).unsqueeze(0)


def predict(lm: LoadedModel, img: Image.Image, top_k: int = 5) -> list[dict]:
    """Return the top-k predictions as [{label, probability}, ...]."""
    x = _preprocess(img, lm.img_size).to(lm.device)
    with torch.no_grad():
        probs = torch.softmax(lm.model(x), dim=1).squeeze(0)
    k = min(top_k, len(lm.classes))
    values, indices = probs.topk(k)
    return [
        {"label": lm.classes[i], "probability": round(float(v), 4)}
        for v, i in zip(values, indices)
    ]


def gradcam_overlay(
    lm: LoadedModel, img: Image.Image, alpha: float = 0.45
) -> Image.Image:
    """Return the input image with a Grad-CAM heatmap blended on top."""
    import matplotlib  # local import: only needed for visualisation

    x = _preprocess(img, lm.img_size).to(lm.device)
    cam_engine = GradCAM(lm.model, get_target_layer(lm.model, lm.backbone))
    heatmap, _ = cam_engine(x)

    base = img.convert("RGB").resize((lm.img_size, lm.img_size))
    base_arr = np.asarray(base, dtype=np.float32) / 255.0

    colored = matplotlib.colormaps["jet"](heatmap)[:, :, :3]  # drop alpha channel
    blended = (1 - alpha) * base_arr + alpha * colored
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
