"""Evaluate the baseline across random and OOD-oriented splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.core.device import get_device
from src.data.dataloader import build_dataloader
from src.data.paddy import build_split_datasets, LABEL_TO_INDEX
from src.data.transforms import build_transforms
from src.evaluation.evaluator import evaluate, predict
from src.evaluation.metrics import compute_confusion_matrix
from src.models.factory import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="experiments/runs/baseline_robustness")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(ROOT / args.config)
    device = get_device(cfg)

    data_cfg = cfg["data"]
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 0))

    eval_transform = build_transforms(image_size, train=False)

    model = build_model(cfg)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    state = checkpoint
    import torch
    payload = torch.load(state, map_location=device)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
    elif isinstance(payload, dict) and "state_dict" in payload:
        model.load_state_dict(payload["state_dict"])
    else:
        model.load_state_dict(payload)

    model.to(device)
    model.eval()

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for split_name in ["random", "variety", "age_old", "age_young"]:
        cfg["data"]["split"] = split_name

        datasets = build_split_datasets(
            cfg,
            train_transform=eval_transform,
            eval_transform=eval_transform,
        )

        for eval_name in ["test", "test_ood"]:
            if eval_name not in datasets:
                continue

            ds = datasets[eval_name]
            loader = build_dataloader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            metrics = evaluate(model, loader, device)
            y_true, y_pred = predict(model, loader, device)

            cm = compute_confusion_matrix(
                y_true,
                y_pred,
                labels=list(range(len(LABEL_TO_INDEX))),
            )

            row = {
                "split": split_name,
                "evaluation": eval_name,
                "n_samples": len(ds),
                **metrics,
            }

            results.append(row)

            with open(
                output_dir / f"{split_name}_{eval_name}_metrics.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(row, f, indent=2)

            with open(
                output_dir / f"{split_name}_{eval_name}_confusion_matrix.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(cm.tolist(), f, indent=2)

            print(
                f"{split_name:10s} | {eval_name:8s} | "
                f"n={len(ds):5d} | "
                f"accuracy={metrics['accuracy']:.4f} | "
                f"macro_f1={metrics['f1_macro']:.4f}"
            )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    import csv

    if results:
        keys = [
            "split",
            "evaluation",
            "n_samples",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]

        with open(
            output_dir / "summary.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

    print(f"\nSaved robustness results to: {output_dir}")


if __name__ == "__main__":
    main()
