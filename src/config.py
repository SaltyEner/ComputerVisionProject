"""Central configuration: paths and default hyper-parameters.

Everything that a script might want to override lives here so the rest of the
code stays free of magic numbers.

Defaults are tuned to be light on a weak laptop CPU:
- a small backbone (mobilenet_v3_small),
- small images (160px),
- a cap on images-per-class,
- and a frozen backbone so training uses the fast "feature caching" path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "plantvillage"     # ImageFolder: one sub-folder per class
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"  # checkpoints + reports
CHECKPOINT_PATH = ARTIFACTS_DIR / "model.pt"

# ImageNet normalisation (the pretrained backbones expect these stats)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    """Hyper-parameters for a training run."""

    backbone: str = "mobilenet_v3_small"   # lightest option; also resnet18 / efficientnet_b0
    img_size: int = 160                    # smaller = much faster on CPU
    batch_size: int = 32
    epochs: int = 40                        # head-only epochs are cheap
    lr: float = 1e-3
    weight_decay: float = 1e-4
    freeze_backbone: bool = True            # True -> fast feature-caching path
    per_class_cap: int = 300                # max images per class (0 = use all)
    val_split: float = 0.2                  # fraction held out for validation
    num_workers: int = 0                    # 0 is safest/fastest on Windows
    seed: int = 42
    patience: int = 4                       # early-stopping patience (epochs)
    data_dir: str = str(DATASET_DIR)
    # filled in at runtime once the dataset is known
    classes: list[str] = field(default_factory=list)


def ensure_dirs() -> None:
    """Create the data/ and artifacts/ folders if they do not exist yet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
