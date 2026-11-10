from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Shared classification metrics. Macro-F1 is the primary project metric."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def compute_confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)
