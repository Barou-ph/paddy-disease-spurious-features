from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def stratified_group_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    label_col: str = "label",
    group_col: str = "group_id",
    n_folds: int = 20,
):
    """
    Train/val/test split that keeps every group in exactly one subset while
    approximately preserving the label distribution.

    The data is cut into `n_folds` stratified group folds; round(test_size *
    n_folds) folds become test, the next round(val_size * n_folds) become val,
    the rest train. With n_folds=20, sizes are multiples of 5%.
    """
    n_test = round(test_size * n_folds)
    n_val = round(val_size * n_folds)
    if n_test < 1 or n_val < 1 or n_test + n_val >= n_folds:
        raise ValueError("val_size/test_size must each give >= 1 fold and leave folds for train.")

    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(folds.split(df, df[label_col], df[group_col])):
        fold_of[idx] = k

    is_test = fold_of < n_test
    is_val = (fold_of >= n_test) & (fold_of < n_test + n_val)
    is_train = ~(is_test | is_val)

    return (
        df[is_train].reset_index(drop=True),
        df[is_val].reset_index(drop=True),
        df[is_test].reset_index(drop=True),
    )


def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.1,
    seed: int = 42,
    label_col: str = "label",
):
    """
    Create train/val/test DataFrames with stratification by label.

    For person/plant/plot/product-related leakage, use group-aware splitting
    instead of this helper.
    """
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1.")

    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=df[label_col],
    )

    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_df[label_col],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_split_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
