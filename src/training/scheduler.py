from __future__ import annotations

import torch


def build_scheduler(optimizer, config: dict):
    cfg = config.get("training", {})
    name = cfg.get("scheduler", "none").lower()

    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg.get("epochs", 30))
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 10)),
            gamma=float(cfg.get("gamma", 0.1)),
        )

    raise ValueError(f"Unsupported scheduler: {name}")
