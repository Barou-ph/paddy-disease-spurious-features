from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


def build_model(config: dict) -> nn.Module:
    """Build a model from the shared experiment config."""
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "efficientnet_b0").lower()
    num_classes = int(model_cfg.get("num_classes", 10))
    pretrained = bool(model_cfg.get("pretrained", True))

    if name != "efficientnet_b0":
        raise ValueError(
            f"Unsupported model '{name}'. Add it to src/models/factory.py "
            "or keep the factory as the shared model entry point."
        )

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model
