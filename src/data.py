"""Dataset & dataloaders for a plant-disease image folder (PlantVillage).

The data is read with torchvision's ``ImageFolder`` so it works with any folder
that has one sub-directory per class, e.g.::

    data/plantvillage/
        Tomato___healthy/        img1.jpg img2.jpg ...
        Tomato___Late_blight/    ...
        Apple___Apple_scab/      ...

To keep training light on a weak CPU we (a) cap the number of images per class
and (b) make a stratified train/validation split in code (PlantVillage ships
without an official split).
"""
from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from . import config


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    """Augmentation for training, plain resize/normalise for eval."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


def _capped_stratified_split(
    targets: list[int], per_class_cap: int, val_split: float, seed: int
) -> tuple[list[int], list[int]]:
    """Return (train_idx, val_idx), capping images per class and splitting."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(targets):
        by_class.setdefault(label, []).append(idx)

    train_idx, val_idx = [], []
    for _label, idxs in by_class.items():
        rng.shuffle(idxs)
        if per_class_cap and len(idxs) > per_class_cap:
            idxs = idxs[:per_class_cap]
        n_val = max(1, int(len(idxs) * val_split))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


def build_dataloaders(
    cfg: config.TrainConfig, augment_train: bool = True
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Return (train_loader, val_loader, class_names).

    When ``augment_train`` is False the training set uses deterministic
    transforms — required by the feature-caching path, where each image is seen
    only once and must always produce the same features.
    """
    config.ensure_dirs()
    data_dir = cfg.data_dir
    try:
        base = ImageFolder(data_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        raise FileNotFoundError(
            f"No dataset found at {data_dir!r}. Download PlantVillage into that "
            "folder (one sub-folder per class) — see the README 'Dataset' section."
        ) from exc

    targets = [label for _, label in base.samples]
    train_idx, val_idx = _capped_stratified_split(
        targets, cfg.per_class_cap, cfg.val_split, cfg.seed
    )

    train_tf = build_transforms(cfg.img_size, train=augment_train)
    val_tf = build_transforms(cfg.img_size, train=False)
    # two ImageFolders over the same directory share the sample ordering, so the
    # split indices line up; they only differ in which transform they apply.
    train_ds = Subset(ImageFolder(data_dir, transform=train_tf), train_idx)
    val_ds = Subset(ImageFolder(data_dir, transform=val_tf), val_idx)

    class_names = [c.replace("___", " · ").replace("_", " ") for c in base.classes]

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader, class_names
