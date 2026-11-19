"""Evaluate a trained baseline checkpoint on a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.checkpoint import load_checkpoint
from src.core.config import load_config
from src.core.device import get_device
from src.core.io import save_json
from src.core.seed import set_seed
from src.data.dataloader import build_dataloader
from src.data.dataset import PaddyImageDataset
from src.data.transforms import build_transforms
from src.evaluation.evaluator import predict
from src.evaluation.metrics import compute_confusion_matrix, compute_metrics
from src.models.factory import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="experiments/runs/evaluation")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg)

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_mapping_path = checkpoint_path.parent / "label_mapping.json"
    if not label_mapping_path.exists():
        raise FileNotFoundError(
            f"Missing {label_mapping_path}. Train the baseline first."
        )

    label_to_index = json.loads(
        label_mapping_path.read_text(encoding="utf-8")
    )

    dataset = PaddyImageDataset(
        args.manifest,
        image_root=cfg["data"].get("image_root"),
        transform=build_transforms(
            int(cfg["data"].get("image_size", 224)),
            train=False,
        ),
        label_to_index=label_to_index,
    )
    loader = build_dataloader(
        dataset,
        batch_size=int(cfg["data"].get("batch_size", 32)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
    )

    model = build_model(cfg)
    load_checkpoint(checkpoint_path, model, device=device)
    model.to(device)

    y_true, y_pred = predict(model, loader, device)
    metrics = compute_metrics(y_true, y_pred)
    cm = compute_confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(label_to_index))),
    )

    labels = [
        label for label, _ in sorted(
            label_to_index.items(),
            key=lambda item: item[1],
        )
    ]

    save_json(output_dir / "metrics.json", metrics)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
