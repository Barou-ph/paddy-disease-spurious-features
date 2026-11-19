"""Train EfficientNet-B0 baseline using the team's leakage-safe splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import clone_config, load_config
from src.core.io import save_history_csv, save_json
from src.core.logger import get_logger
from src.core.reproducibility import collect_environment
from src.core.seed import set_seed
from src.core.device import get_device
from src.data.dataloader import build_dataloader
from src.data.paddy import build_split_datasets, LABEL_TO_INDEX
from src.data.transforms import build_transforms
from src.evaluation.evaluator import evaluate, predict
from src.evaluation.metrics import compute_confusion_matrix
from src.models.factory import build_model
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--split",
        choices=["random", "variety", "age_old", "age_young"],
        help="Override data.split in the config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(ROOT / args.config)

    if args.output_dir:
        cfg.setdefault("output", {})["checkpoint_dir"] = args.output_dir
    if args.device:
        cfg["device"] = args.device
    if args.split:
        cfg.setdefault("data", {})["split"] = args.split

    set_seed(
        int(cfg["seed"]),
        deterministic=bool(cfg.get("deterministic", True)),
    )
    device = get_device(cfg)

    output_dir = Path(cfg["output"]["checkpoint_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("baseline", output_dir / "train.log")
    logger.info("Starting baseline experiment")
    logger.info("Device: %s", device)

    data_cfg = cfg["data"]
    image_size = int(data_cfg.get("image_size", 224))

    train_transform = build_transforms(
        image_size,
        train=True,
        horizontal_flip=bool(data_cfg.get("horizontal_flip", True)),
        rotation_degrees=float(data_cfg.get("rotation_degrees", 10)),
    )
    eval_transform = build_transforms(image_size, train=False)

    datasets = build_split_datasets(
        cfg,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    required = {"train", "val", "test"}
    missing = required - set(datasets)
    if missing:
        raise ValueError(f"Split is missing required subsets: {sorted(missing)}")

    train_ds = datasets["train"]
    val_ds = datasets["val"]
    test_ds = datasets["test"]

    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 0))

    train_loader = build_dataloader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
    )
    val_loader = build_dataloader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    test_loader = build_dataloader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    logger.info(
        "Split=%s | train=%d | val=%d | test=%d",
        data_cfg.get("split", "random"),
        len(train_ds), len(val_ds), len(test_ds),
    )

    model = build_model(cfg)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=cfg,
        scheduler=scheduler,
        logger=logger,
    )

    history, best_epoch, best_metric = trainer.fit(
        train_loader,
        val_loader,
    )

    save_history_csv(output_dir / "history.csv", history)
    save_json(output_dir / "config.json", clone_config(cfg))
    save_json(output_dir / "environment.json", collect_environment())
    save_json(output_dir / "label_mapping.json", LABEL_TO_INDEX)

    from src.core.checkpoint import load_checkpoint

    checkpoint_path = output_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Training finished but best checkpoint was not found: {checkpoint_path}"
        )

    load_checkpoint(checkpoint_path, model, device=device)

    test_metrics = evaluate(model, test_loader, device)
    y_true, y_pred = predict(model, test_loader, device)

    class_indices = list(range(len(LABEL_TO_INDEX)))
    cm = compute_confusion_matrix(
        y_true,
        y_pred,
        labels=class_indices,
    )

    class_names = list(LABEL_TO_INDEX.keys())
    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(output_dir / "confusion_matrix.csv")

    test_metrics["best_epoch"] = int(best_epoch)
    test_metrics["best_val_f1_macro"] = float(best_metric)

    save_json(output_dir / "test_metrics.json", test_metrics)

    summary = {
        "experiment": "baseline",
        "split": data_cfg.get("split", "random"),
        "best_epoch": int(best_epoch),
        "best_val_f1_macro": float(best_metric),
        "test_metrics": test_metrics,
        "device": str(device),
        "num_classes": len(LABEL_TO_INDEX),
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "checkpoint": str(checkpoint_path),
    }
    save_json(output_dir / "summary.json", summary)

    logger.info("Best epoch: %d", best_epoch)
    logger.info("Best validation macro-F1: %.5f", best_metric)
    logger.info("Test metrics: %s", test_metrics)
    logger.info("Baseline experiment complete.")


if __name__ == "__main__":
    main()
