from __future__ import annotations

import torch.nn as nn


def build_loss(config: dict) -> nn.Module:
    name = config.get("training", {}).get("loss", "cross_entropy").lower()

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()

    raise ValueError(f"Unsupported loss: {name}")
