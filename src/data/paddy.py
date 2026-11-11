"""Paddy Doctor specifics: the fixed class order and the team's shared splits.

The split CSVs in data/splits/ are produced by scripts/prepare_splits.py and
committed, so every experiment trains and evaluates on identical data.

Typical use:

    from src.data.paddy import build_split_datasets
    datasets = build_split_datasets(config)       # config["data"]["split"] picks the split
    train_ds, val_ds = datasets["train"], datasets["val"]
    test_ds = datasets["test"]                    # in-distribution test
    ood_ds = datasets.get("test_ood")             # held-out domain, if the split has one
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from src.data.dataset import PaddyImageDataset
from src.data.transforms import build_transforms

ROOT = Path(__file__).resolve().parents[2]

# Fixed order. Always pass LABEL_TO_INDEX to the dataset: some evaluation
# subsets (e.g. age_old/test_ood) do not contain every class, and inferring
# the mapping from the labels present would silently shift the indices.
CLASSES = (
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
)
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(CLASSES)}

SPLIT_NAMES = ("random", "variety", "age_old", "age_young")
DEFAULT_IMAGE_ROOT = "data/raw/paddy-disease-classification/train_images"
DEFAULT_SPLIT_DIR = "data/splits"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_split(name: str = "random", split_dir: str | Path = DEFAULT_SPLIT_DIR) -> dict[str, pd.DataFrame]:
    """Return {"train", "val", "test"[, "test_ood"]} DataFrames for one split."""
    folder = _resolve(split_dir) / name
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Split '{name}' not found in {folder.parent}. Available: {SPLIT_NAMES}. "
            "Run `python scripts/prepare_splits.py` if data/splits/ is missing."
        )
    return {path.stem: pd.read_csv(path) for path in sorted(folder.glob("*.csv"))}


def build_split_datasets(
    config: dict,
    train_transform: Callable | None = None,
    eval_transform: Callable | None = None,
) -> dict[str, PaddyImageDataset]:
    """Build one dataset per subset of the split named in config["data"]["split"].

    Reads data.split (default "random"), data.split_dir, data.image_root and
    data.image_size. Only the "train" subset gets the training transform.
    """
    data_cfg = config.get("data", {})
    image_size = int(data_cfg.get("image_size", 224))
    train_transform = train_transform or build_transforms(image_size, train=True)
    eval_transform = eval_transform or build_transforms(image_size, train=False)
    image_root = _resolve(data_cfg.get("image_root", DEFAULT_IMAGE_ROOT))

    frames = load_split(
        data_cfg.get("split", "random"), data_cfg.get("split_dir", DEFAULT_SPLIT_DIR)
    )
    return {
        subset: PaddyImageDataset(
            df,
            image_root=image_root,
            transform=train_transform if subset == "train" else eval_transform,
            label_to_index=LABEL_TO_INDEX,
        )
        for subset, df in frames.items()
    }
