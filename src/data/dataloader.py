from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def build_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool | None = None,
    drop_last: bool = False,
):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    # persistent_workers requires num_workers > 0.
    if num_workers > 0:
        kwargs["persistent_workers"] = True

    return DataLoader(**kwargs)
