"""Download a light subset of PlantVillage straight from the public GitHub mirror.

No Kaggle account needed. Folder listings come from the GitHub API (a handful of
requests), the images themselves from the raw CDN (not rate-limited).

Usage:
    python -m src.get_data                       # default classes, 100 imgs each
    python -m src.get_data --per-class 60
    python -m src.get_data --classes Tomato___healthy Tomato___Late_blight
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

from . import config

API = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"
REF = "?ref=master"
_HEADERS = {"User-Agent": "leafdoctor-downloader"}

# a compact, story-friendly default subset (healthy + diseased, 3 crops)
DEFAULT_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Apple___healthy",
    "Apple___Apple_scab",
]


def _get_json(url: str):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        dest.write_bytes(resp.read())


def fetch_class(cls: str, per_class: int, out_root: Path) -> int:
    listing = _get_json(f"{API}/{urllib.parse.quote(cls)}{REF}")
    images = [
        item for item in listing
        if item["type"] == "file" and item["name"].lower().endswith((".jpg", ".jpeg", ".png"))
    ][:per_class]

    cls_dir = out_root / cls
    cls_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for item in images:
        dest = cls_dir / item["name"]
        if dest.exists():
            saved += 1
            continue
        try:
            _download(item["download_url"], dest)
            saved += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ! skip {item['name']}: {exc}")
    print(f"  {cls}: {saved} images")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a PlantVillage subset.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--out", default=str(config.DATASET_DIR))
    args = parser.parse_args()

    out_root = Path(args.out)
    print(f"Downloading {len(args.classes)} classes "
          f"(<= {args.per_class} imgs each) into {out_root}")
    total = 0
    for cls in args.classes:
        total += fetch_class(cls, args.per_class, out_root)
    print(f"Done. {total} images across {len(args.classes)} classes -> {out_root}")


if __name__ == "__main__":
    main()
