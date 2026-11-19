from __future__ import annotations

import torch


def build_optimizer(model, config: dict):
    cfg = config.get("training", {})
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("learning_rate", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {name}")
