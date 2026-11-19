"""Create a reproducible stratified train/val/test split from a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.split import save_split_manifests, stratified_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="data/processed/random_split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.1)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    train_df, val_df, test_df = stratified_split(
        df,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    save_split_manifests(
        train_df,
        val_df,
        test_df,
        Path(args.output_dir),
    )

    print(f"train={len(train_df)}")
    print(f"val={len(val_df)}")
    print(f"test={len(test_df)}")
