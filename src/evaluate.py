"""Evaluate a trained checkpoint on the test set.

Produces a classification report (precision/recall/F1 per class) and saves a
confusion-matrix figure to artifacts/confusion_matrix.png.

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from . import config
from .data import build_dataloaders
from .inference import load_model


def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            y_pred.extend(outputs.argmax(1).cpu().tolist())
            y_true.extend(targets.tolist())
    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main() -> None:
    lm = load_model()
    cfg = config.TrainConfig(
        backbone=lm.backbone, img_size=lm.img_size, num_workers=2
    )
    _, test_loader, _ = build_dataloaders(cfg)

    y_true, y_pred = collect_predictions(lm.model, test_loader, lm.device)
    print(classification_report(y_true, y_pred, target_names=lm.classes, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, lm.classes, config.ARTIFACTS_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
