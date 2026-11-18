"""Build the cleaned manifest and the fixed, leakage-safe splits shared by the team.

Run once after downloading the data (see data/README.md):

    python scripts/prepare_splits.py

Outputs (committed, so every experiment uses identical splits):

    data/manifest.csv                          every train.csv image + group_id + kept/drop_reason
    data/splits/<split>/{train,val,test}.csv   in-distribution train / val / test
    data/splits/<split>/test_ood.csv           held-out domain (variety / age splits only)
    data/splits/summary.json, summary.md       subset sizes, class coverage, leakage checks

Cleaning (decided from the EDA in reports/eda/):
  * exact duplicates (same file bytes): keep the lowest image_id, drop the rest;
  * label conflicts: images whose dHash is within --conflict-dist bits of an image
    with a different label are the same photo labelled twice -> both dropped;
  * group_id: connected components of the near-duplicate graph (dHash distance
    <= --group-dist). A group never straddles two subsets, so re-shots of one
    scene cannot leak from train into val/test.

Splits (all stratified by label and grouped by group_id, 70/15/15 in-domain):
  random     no domain shift; the reference in-distribution setting
  variety    in-domain ADT45 (67% of images, all 10 classes); test_ood = other varieties
  age_old    in-domain age <= 70 days; test_ood = age >= 72 (later growth stage)
  age_young  in-domain age >= 55 days; test_ood = age <= 50 (earlier growth stage)

Each row also carries meta_pred / meta_aligned: the label a lookup table on
(variety, age) -- fitted on that split's train subset only -- would predict.
Rows where meta_aligned is False are "shortcut-conflicting": metadata alone
points to the wrong class, so a model leaning on variety/age cues should do
worse on them.
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.dedup import connected_groups, hamming_pairs, hashes_to_u64, probe_images
from src.data.paddy import CLASSES
from src.data.split import stratified_group_split

OUTPUT_COLUMNS = [
    "image_id", "image_path", "label", "variety", "age", "group_id",
    "split", "meta_pred", "meta_aligned",
]

# name -> (in-domain mask, description). None = no held-out domain.
SPLITS = {
    "random": (None, "stratified group split of all clean images"),
    "variety": (lambda df: df["variety"] == "ADT45",
                "in-domain ADT45; test_ood = the other 9 varieties"),
    "age_old": (lambda df: df["age"] <= 70,
                "in-domain age <= 70 days; test_ood = age >= 72 days"),
    "age_young": (lambda df: df["age"] >= 55,
                  "in-domain age >= 55 days; test_ood = age <= 50 days"),
}


# ----------------------------------------------------------------------------- hashing
def load_hashes(df: pd.DataFrame, image_root: Path, cache: Path, workers: int) -> pd.DataFrame:
    if cache.exists():
        cached = pd.read_csv(cache, dtype={"dhash": str})
        if set(df["image_id"]) <= set(cached["image_id"]):
            return cached
    paths = [image_root / p for p in df["image_path"]]
    probes = probe_images(paths, workers, progress=partial(tqdm, total=len(paths), desc="hashing"))
    probes.insert(0, "image_id", df["image_id"].values)
    cache.parent.mkdir(parents=True, exist_ok=True)
    probes.to_csv(cache, index=False)
    return probes


# ----------------------------------------------------------------------------- cleaning
def build_manifest(df: pd.DataFrame, group_dist: int, conflict_dist: int) -> tuple[pd.DataFrame, dict]:
    df = df.sort_values("image_id").reset_index(drop=True)
    pairs = hamming_pairs(hashes_to_u64(df["dhash"]), group_dist)

    groups = connected_groups(len(df), pairs)
    # Renumber components by first appearance so ids are stable and readable.
    order = {g: k for k, g in enumerate(pd.unique(groups))}
    df["group_id"] = [f"g{order[g]:05d}" for g in groups]

    labels = df["label"].to_numpy()
    close = pairs[pairs["dist"] <= conflict_dist]
    cross = close[labels[close["i"]] != labels[close["j"]]]
    conflict_idx = sorted(set(cross["i"]) | set(cross["j"]))

    df["drop_reason"] = ""
    df.loc[conflict_idx, "drop_reason"] = "label_conflict"
    extra_copy = df.duplicated("md5", keep="first") & (df["drop_reason"] == "")
    df.loc[extra_copy, "drop_reason"] = "exact_duplicate"

    group_sizes = df["group_id"].value_counts()
    stats = {
        "n_images": int(len(df)),
        "n_dropped_label_conflict": int((df["drop_reason"] == "label_conflict").sum()),
        "n_dropped_exact_duplicate": int((df["drop_reason"] == "exact_duplicate").sum()),
        "n_clean": int((df["drop_reason"] == "").sum()),
        "n_groups": int(len(group_sizes)),
        "n_groups_with_2plus_images": int((group_sizes > 1).sum()),
        "largest_group": int(group_sizes.max()),
        "n_label_conflict_pairs": int(len(cross)),
    }
    return df, stats


# ----------------------------------------------------------------------------- shortcut lookup
def _majority(labels: pd.Series) -> str:
    counts = labels.value_counts().sort_index()  # alphabetical tie-break, deterministic
    return str(counts.idxmax())


def add_metadata_prediction(train: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> None:
    """Fit (variety, age) -> majority label on train; back off to age, then global."""
    by_variety_age = train.groupby(["variety", "age"])["label"].agg(_majority).to_dict()
    by_age = train.groupby("age")["label"].agg(_majority).to_dict()
    fallback = _majority(train["label"])

    for frame in frames.values():
        frame["meta_pred"] = [
            by_variety_age.get((v, a), by_age.get(a, fallback))
            for v, a in zip(frame["variety"], frame["age"])
        ]
        frame["meta_aligned"] = frame["meta_pred"] == frame["label"]


# ----------------------------------------------------------------------------- splitting
def make_split(clean: pd.DataFrame, in_domain, seed: int) -> tuple[dict[str, pd.DataFrame], int]:
    dropped_ood = 0
    if in_domain is None:
        pool, ood = clean, None
    else:
        mask = in_domain(clean)
        pool, ood = clean[mask], clean[~mask]
        # A group with members on both sides: its out-of-domain members are near
        # copies of in-domain images, so they would inflate OOD scores. Drop them.
        shared = set(pool["group_id"]) & set(ood["group_id"])
        dropped_ood = int(ood["group_id"].isin(shared).sum())
        ood = ood[~ood["group_id"].isin(shared)].reset_index(drop=True)

    train, val, test = stratified_group_split(pool, val_size=0.15, test_size=0.15, seed=seed)
    frames = {"train": train, "val": val, "test": test}
    if ood is not None:
        frames["test_ood"] = ood
    for name, frame in frames.items():
        frame["split"] = name
    add_metadata_prediction(train, frames)
    return frames, dropped_ood


def check_no_leakage(frames: dict[str, pd.DataFrame]) -> None:
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(frames[a]["group_id"]) & set(frames[b]["group_id"]):
                raise AssertionError(f"group_id shared between {a} and {b}")
            if set(frames[a]["image_id"]) & set(frames[b]["image_id"]):
                raise AssertionError(f"image_id shared between {a} and {b}")


def describe(frame: pd.DataFrame) -> dict:
    counts = frame["label"].value_counts()
    return {
        "n": int(len(frame)),
        "class_counts": {c: int(counts.get(c, 0)) for c in CLASSES},
        "missing_classes": [c for c in CLASSES if counts.get(c, 0) == 0],
        "meta_aligned_frac": float(frame["meta_aligned"].mean()),
        "varieties": int(frame["variety"].nunique()),
        "age_range": [int(frame["age"].min()), int(frame["age"].max())],
    }


def summary_markdown(summary: dict) -> str:
    c = summary["cleaning"]
    lines = [
        "# Split summary",
        "",
        "Generated by `scripts/prepare_splits.py` -- do not edit by hand.",
        "",
        f"- Images in train.csv: **{c['n_images']}**; dropped as label conflicts: "
        f"**{c['n_dropped_label_conflict']}** ({c['n_label_conflict_pairs']} cross-label pairs); "
        f"dropped as exact duplicates: **{c['n_dropped_exact_duplicate']}**; clean: **{c['n_clean']}**",
        f"- Groups (dHash <= {summary['group_dist']}): {c['n_groups']} "
        f"({c['n_groups_with_2plus_images']} with 2+ images, largest {c['largest_group']})",
        f"- Seed {summary['seed']}; no group_id or image_id is shared between subsets of a split.",
        "",
        "`meta-aligned` = share of rows where the (variety, age) lookup fitted on train "
        "predicts the true label.",
        "",
        "| split | subset | images | missing classes | smallest class | meta-aligned | varieties | ages |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, info in summary["splits"].items():
        for subset, d in info["subsets"].items():
            present = [v for v in d["class_counts"].values() if v > 0]
            missing = ", ".join(d["missing_classes"]) or "-"
            lines.append(
                f"| {name} | {subset} | {d['n']} | {missing} | {min(present)} | "
                f"{100 * d['meta_aligned_frac']:.1f}% | {d['varieties']} | "
                f"{d['age_range'][0]}-{d['age_range'][1]} |"
            )
    lines += ["", "Descriptions:", ""]
    for name, info in summary["splits"].items():
        extra = (f" ({info['dropped_ood_near_copies']} out-of-domain near copies of in-domain "
                 "images removed)" if info["dropped_ood_near_copies"] else "")
        lines.append(f"- `{name}`: {info['description']}{extra}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/paddy-disease-classification")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--hash-cache", type=Path, default=ROOT / "data/processed/image_hashes.csv")
    ap.add_argument("--group-dist", type=int, default=8,
                    help="dHash bits; pairs at <= this distance share a group (default 8/64)")
    ap.add_argument("--conflict-dist", type=int, default=4,
                    help="dHash bits; cross-label pairs at <= this distance are dropped (default 4/64)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    raw = pd.read_csv(args.data_dir / "train.csv")
    raw["image_path"] = raw["label"] + "/" + raw["image_id"]
    unknown = set(raw["label"]) - set(CLASSES)
    if unknown:
        raise SystemExit(f"Unexpected labels in train.csv: {sorted(unknown)}")

    hashes = load_hashes(raw, args.data_dir / "train_images", args.hash_cache, args.workers)
    df = raw.merge(hashes[["image_id", "md5", "dhash"]], on="image_id", how="left", validate="1:1")
    manifest, cleaning = build_manifest(df, args.group_dist, args.conflict_dist)

    out = args.out_dir
    manifest["kept"] = manifest["drop_reason"] == ""
    manifest[["image_id", "image_path", "label", "variety", "age", "group_id", "kept",
              "drop_reason"]].to_csv(out / "manifest.csv", index=False)

    clean = manifest[manifest["drop_reason"] == ""].reset_index(drop=True)
    summary = {"seed": args.seed, "group_dist": args.group_dist,
               "conflict_dist": args.conflict_dist, "cleaning": cleaning, "splits": {}}

    for name, (in_domain, description) in SPLITS.items():
        frames, dropped_ood = make_split(clean, in_domain, args.seed)
        check_no_leakage(frames)
        folder = out / "splits" / name
        folder.mkdir(parents=True, exist_ok=True)
        for subset, frame in frames.items():
            frame[OUTPUT_COLUMNS].to_csv(folder / f"{subset}.csv", index=False)
        summary["splits"][name] = {
            "description": description,
            "dropped_ood_near_copies": dropped_ood,
            "subsets": {subset: describe(frame) for subset, frame in frames.items()},
        }

    (out / "splits" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "splits" / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    print(summary_markdown(summary))


if __name__ == "__main__":
    main()
