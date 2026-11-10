from __future__ import annotations

from torch.utils.data import DataLoader


def build_dataloader(dataset, batch_size: int = 32, shuffle: bool = False, num_workers: int = 4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
