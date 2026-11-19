
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute classification metrics over classes present in ground truth."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    present_labels = np.unique(y_true)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=present_labels,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def compute_confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)
