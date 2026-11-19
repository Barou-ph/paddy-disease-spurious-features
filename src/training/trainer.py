from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from src.core.checkpoint import save_checkpoint
from src.evaluation.evaluator import evaluate


class Trainer:
    """Shared image-classification training loop."""

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

        training_cfg = config.get("training", {})
        self.epochs = int(training_cfg.get("epochs", 30))
        self.grad_clip_norm = training_cfg.get("grad_clip_norm")
        self.early_stopping_patience = training_cfg.get("early_stopping_patience")

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

                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(self.grad_clip_norm),
                    )

                self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

        return total_loss / max(total_items, 1)

    def fit(self, train_loader, val_loader=None, metric_fn: Callable | None = None):
        best_metric = float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0
        history = []

        primary_metric = self.config.get("evaluation", {}).get(
            "primary_metric", "f1_macro"
        )

        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)

            row = {"epoch": epoch, "train_loss": train_loss}

            if val_loader is not None:
                with torch.no_grad():
                    val_loss = self._run_epoch(val_loader, train=False)

                val_metrics = (
                    metric_fn(self.model, val_loader, self.device)
                    if metric_fn
                    else evaluate(self.model, val_loader, self.device)
                )
                row["val_loss"] = val_loss
                row.update({f"val_{k}": v for k, v in val_metrics.items()})
                current_metric = float(
                    val_metrics.get(primary_metric, float("-inf"))
                )
            else:
                current_metric = -train_loss

            if self.scheduler is not None:
                self.scheduler.step()

            row["learning_rate"] = self.optimizer.param_groups[0]["lr"]
            history.append(row)

            if self.logger:
                self.logger.info(
                    "epoch=%d train_loss=%.5f val_loss=%s val_%s=%s",
                    epoch,
                    train_loss,
                    f"{row['val_loss']:.5f}" if "val_loss" in row else "n/a",
                    primary_metric,
                    f"{current_metric:.5f}",
                )

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                epochs_without_improvement = 0

                save_checkpoint(
                    self.checkpoint_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch=epoch,
                    metrics=row,
                    config=self.config,
                )
            else:
                epochs_without_improvement += 1

            if (
                self.early_stopping_patience is not None
                and epochs_without_improvement >= int(self.early_stopping_patience)
            ):
                if self.logger:
                    self.logger.info(
                        "Early stopping at epoch %d; best epoch=%d.",
                        epoch,
                        best_epoch,
                    )
                break

        save_checkpoint(
            self.checkpoint_dir / "last.pt",
            self.model,
            self.optimizer,
            self.scheduler,
            epoch=history[-1]["epoch"] if history else None,
            metrics=history[-1] if history else {},
            config=self.config,
        )

        return history, best_epoch, best_metric
