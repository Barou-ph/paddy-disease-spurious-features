from __future__ import annotations

import torch


def get_device(config: dict | None = None) -> torch.device:
    """Select CUDA when available unless config explicitly requests another device."""
    requested = None
    if config:
        requested = config.get("device") or config.get("hardware", {}).get("device")

    if requested and str(requested).lower() != "auto":
        return torch.device(requested)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
