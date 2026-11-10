"""Minimal shared-core import/config smoke test."""

from pathlib import Path
import sys

# Add project root to Python import path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_config, get
from src.core.device import get_device
from src.core.reproducibility import collect_environment
from src.models.factory import build_model


def main():
    config = load_config(ROOT / "configs" / "baseline.yaml")

    assert get(config, "model.name") == "efficientnet_b0"
    assert get(config, "evaluation.primary_metric") == "f1_macro"

    device = get_device(config)
    model = build_model(config).to(device)

    print("Shared core smoke test: OK")
    print(f"device={device}")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"environment={collect_environment()}")


if __name__ == "__main__":
    main()
