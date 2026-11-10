from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from src.core.checkpoint import save_checkpoint
from src.evaluation.evaluator import evaluate


class Trainer:
    """Shared classification training loop.

    Research-specific methods should plug into this interface rather than
    duplicating the complete training loop.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        config: dict,
        scheduler=None,
        logger=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.logger = logger

        self.epochs = int(config.get("training", {}).get("epochs", 30))
        self.checkpoint_dir = Path(
            config.get("output", {}).get("checkpoint_dir", "checkpoints")
        )

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        total_items = 0

        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            if train:
                loss.backward()
                self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

        return total_loss / max(total_items, 1)

    def fit(self, train_loader, val_loader=None, metric_fn: Callable | None = None):
        best_metric = float("-inf")
        history = []

        primary_metric = self.config.get("evaluation", {}).get(
            "primary_metric", "f1_macro"
        )

        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
            }

            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, train=False)
                val_metrics = (
                    metric_fn(self.model, val_loader, self.device)
                    if metric_fn
                    else evaluate(self.model, val_loader, self.device)
                )
                row["val_loss"] = val_loss
                row.update({f"val_{k}": v for k, v in val_metrics.items()})
                current_metric = val_metrics.get(primary_metric, float("-inf"))
            else:
                current_metric = -train_loss

            if self.scheduler is not None:
                self.scheduler.step()

            history.append(row)

            if self.logger:
                self.logger.info(row)

            if current_metric > best_metric:
                best_metric = current_metric
                save_checkpoint(
                    self.checkpoint_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch=epoch,
                    metrics=row,
                    config=self.config,
                )

        return history
