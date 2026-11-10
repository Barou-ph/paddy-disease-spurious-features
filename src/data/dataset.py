from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PaddyImageDataset(Dataset):
    """
    Dataset for a manifest CSV.

    Required columns:
        image_path, label

    Optional metadata columns:
        variety, age, split, group_id

    image_path is resolved relative to image_root when it is not absolute.
    """

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        image_root: str | Path | None = None,
        transform: Callable | None = None,
        label_to_index: dict[str, int] | None = None,
    ):
        if isinstance(manifest, (str, Path)):
            self.df = pd.read_csv(manifest).copy()
        else:
            self.df = manifest.copy()

        required = {"image_path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        self.image_root = Path(image_root) if image_root else None
        self.transform = transform

        labels = self.df["label"].astype(str)
        if label_to_index is None:
            classes = sorted(labels.unique())
            self.label_to_index = {name: idx for idx, name in enumerate(classes)}
        else:
            self.label_to_index = dict(label_to_index)

        self.classes = [
            label for label, _ in sorted(self.label_to_index.items(), key=lambda x: x[1])
        ]

    def __len__(self) -> int:
        return len(self.df)

    def _image_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or self.image_root is None:
            return path
        return self.image_root / path

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(self._image_path(str(row["image_path"]))).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.label_to_index[str(row["label"])]
        return {
            "image": image,
            "label": label,
            "index": index,
        }
