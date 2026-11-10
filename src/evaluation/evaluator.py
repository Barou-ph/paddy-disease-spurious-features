from __future__ import annotations

import torch

from src.evaluation.metrics import compute_metrics


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()

    y_true = []
    y_pred = []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]

        logits = model(images)
        predictions = logits.argmax(dim=1).cpu()

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.tolist())

    return y_true, y_pred


def evaluate(model, dataloader, device) -> dict[str, float]:
    y_true, y_pred = predict(model, dataloader, device)
    return compute_metrics(y_true, y_pred)
