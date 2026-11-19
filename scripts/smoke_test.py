"""Minimal shared-core smoke test. It never downloads pretrained weights."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import clone_config, get, load_config
from src.core.device import get_device
from src.core.reproducibility import collect_environment
from src.models.factory import build_model


def main():
    config = load_config(ROOT / "configs" / "baseline.yaml")
    smoke_config = clone_config(config)
    smoke_config["model"]["pretrained"] = False

    assert get(config, "model.name") == "efficientnet_b0"
    assert get(config, "evaluation.primary_metric") == "f1_macro"

    device = get_device(config)
    model = build_model(smoke_config).to(device)

    print("Shared core smoke test: OK")
    print(f"device={device}")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"environment={collect_environment()}")


if __name__ == "__main__":
    main()
