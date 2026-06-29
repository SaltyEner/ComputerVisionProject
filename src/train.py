"""Training entry point — light by default, made for a weak CPU.

With a frozen backbone (the default) it uses the **feature caching** path:
each image goes through the backbone exactly once, the resulting feature vectors
are cached in memory, and only a tiny linear classifier is trained on them — so
every epoch after the first is essentially instant.

Usage:
    python -m src.train                       # mobilenet, frozen, fast
    python -m src.train --per-class-cap 150   # even lighter
    python -m src.train --backbone resnet18 --unfreeze --epochs 12   # full (GPU)
"""
from __future__ import annotations

import argparse
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import config
from .data import build_dataloaders
from .model import (
    build_model,
    get_classifier,
    get_device,
    set_classifier,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_features(
    feature_model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the (head-less) backbone once over a loader and cache the features."""
    feature_model.eval()
    feats, labels = [], []
    for images, targets in tqdm(loader, desc="caching features", leave=False):
        feats.append(feature_model(images.to(device)).cpu())
        labels.append(targets)
    return torch.cat(feats), torch.cat(labels)


def train_light(cfg: config.TrainConfig) -> float:
    """Frozen backbone + cached features + linear head. Fast on CPU."""
    device = get_device()
    # deterministic transforms: each image is cached once, so no random augmentation
    train_loader, val_loader, classes = build_dataloaders(cfg, augment_train=False)
    cfg.classes = classes
    print(f"{len(classes)} classes | "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val")

    model = build_model(
        num_classes=len(classes),
        backbone=cfg.backbone,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    feat_dim = get_classifier(model, cfg.backbone).in_features
    set_classifier(model, cfg.backbone, nn.Identity())  # backbone now outputs features

    print("Extracting features (one pass over the data)...")
    x_train, y_train = extract_features(model, train_loader, device)
    x_val, y_val = extract_features(model, val_loader, device)

    # train a small linear classifier on the cached vectors — this is the cheap part
    head = nn.Linear(feat_dim, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    feat_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=256, shuffle=True
    )

    best_acc, best_state, no_improve = 0.0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        head.train()
        for xb, yb in feat_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(head(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_acc = (head(x_val.to(device)).argmax(1).cpu() == y_val).float().mean()
        val_acc = float(val_acc)
        print(f"Epoch {epoch:2d}/{cfg.epochs} | val acc {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc, best_state, no_improve = val_acc, copy.deepcopy(head.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stopping (no improvement for {cfg.patience} epochs).")
                break

    head.load_state_dict(best_state)
    set_classifier(model, cfg.backbone, head.cpu())  # re-attach the trained head
    _save_checkpoint(model, cfg, best_acc)
    return best_acc


def _run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(is_train):
        for images, targets in tqdm(loader, leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += images.size(0)
    return loss_sum / total, correct / total


def train_full(cfg: config.TrainConfig) -> float:
    """Fine-tune the whole backbone. Heavier — use a GPU (Colab)."""
    device = get_device()
    train_loader, val_loader, classes = build_dataloaders(cfg, augment_train=True)
    cfg.classes = classes
    print(f"{len(classes)} classes | "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val")

    model = build_model(len(classes), cfg.backbone, True, freeze_backbone=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_acc, no_improve = 0.0, 0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:2d}/{cfg.epochs} | train acc {tr_acc:.3f} | "
              f"val acc {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc, no_improve = va_acc, 0
            _save_checkpoint(model, cfg, best_acc)
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print("Early stopping.")
                break
    return best_acc


def _save_checkpoint(model: nn.Module, cfg: config.TrainConfig, acc: float) -> None:
    config.ensure_dirs()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": cfg.classes,
            "backbone": cfg.backbone,
            "img_size": cfg.img_size,
            "val_acc": acc,
        },
        config.CHECKPOINT_PATH,
    )
    (config.ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps({"best_val_acc": acc, "backbone": cfg.backbone}, indent=2)
    )


def main() -> None:
    d = config.TrainConfig()
    parser = argparse.ArgumentParser(description="Train the plant-disease classifier.")
    parser.add_argument("--backbone", default=d.backbone)
    parser.add_argument("--epochs", type=int, default=d.epochs)
    parser.add_argument("--batch-size", type=int, default=d.batch_size)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--img-size", type=int, default=d.img_size)
    parser.add_argument("--per-class-cap", type=int, default=d.per_class_cap,
                        help="Max images per class (0 = use all).")
    parser.add_argument("--data-dir", default=d.data_dir)
    parser.add_argument("--unfreeze", action="store_true",
                        help="Fine-tune the whole backbone (needs a GPU).")
    args = parser.parse_args()

    cfg = config.TrainConfig(
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        per_class_cap=args.per_class_cap,
        data_dir=args.data_dir,
        freeze_backbone=not args.unfreeze,
    )
    set_seed(cfg.seed)
    print(f"Device: {get_device()} | backbone: {cfg.backbone} | "
          f"mode: {'full fine-tune' if args.unfreeze else 'light (feature caching)'}")

    best = train_full(cfg) if args.unfreeze else train_light(cfg)
    print(f"Best val accuracy: {best:.3f} -> {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
