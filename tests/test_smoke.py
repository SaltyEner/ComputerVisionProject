"""Smoke tests that run on CPU in seconds, without the real dataset.

They check that:
- every backbone builds and produces the right output shape,
- Grad-CAM returns a normalised heatmap,
- the full light training path (feature caching -> linear head -> checkpoint ->
  predict -> Grad-CAM overlay) works end-to-end on a tiny synthetic dataset.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src import config
from src.gradcam import GradCAM
from src.model import build_model, get_target_layer

BACKBONES = ["mobilenet_v3_small", "resnet18", "efficientnet_b0"]


@pytest.mark.parametrize("backbone", BACKBONES)
def test_forward_shape(backbone: str) -> None:
    model = build_model(7, backbone=backbone, pretrained=False)
    out = model(torch.randn(2, 3, 160, 160))
    assert out.shape == (2, 7)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_gradcam_heatmap_is_normalised(backbone: str) -> None:
    model = build_model(7, backbone=backbone, pretrained=False)
    cam = GradCAM(model, get_target_layer(model, backbone))
    heatmap, idx = cam(torch.randn(1, 3, 160, 160))
    assert heatmap.shape == (160, 160)
    assert 0 <= idx < 7
    assert np.isclose(heatmap.min(), 0.0)
    assert heatmap.max() <= 1.0 + 1e-6


def test_unknown_backbone_raises() -> None:
    with pytest.raises(ValueError):
        build_model(7, backbone="not_a_model", pretrained=False)


def _make_fake_dataset(root, n_classes=3, per_class=6, size=64) -> None:
    """Create a tiny ImageFolder of random images so training has data to chew on."""
    rng = np.random.default_rng(0)
    for c in range(n_classes):
        cls_dir = root / f"Plant{c}___disease"
        cls_dir.mkdir(parents=True)
        for i in range(per_class):
            arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img{i}.jpg")


def test_light_training_end_to_end(tmp_path, monkeypatch) -> None:
    from src import train
    from src.inference import gradcam_overlay, load_model, predict

    data_dir = tmp_path / "plantvillage"
    _make_fake_dataset(data_dir)
    # redirect artifacts to a temp folder so we don't touch the real one
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "CHECKPOINT_PATH", tmp_path / "artifacts" / "model.pt")

    cfg = config.TrainConfig(
        data_dir=str(data_dir), img_size=64, epochs=2,
        per_class_cap=0, batch_size=8,
    )
    train.train_light(cfg)
    assert config.CHECKPOINT_PATH.exists()

    lm = load_model(config.CHECKPOINT_PATH)
    img = Image.fromarray(
        np.random.default_rng(1).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    )
    preds = predict(lm, img, top_k=3)
    assert len(preds) == 3
    assert abs(sum(p["probability"] for p in preds) - 1.0) < 0.5  # softmax-ish
    overlay = gradcam_overlay(lm, img)
    assert overlay.size == (64, 64)
