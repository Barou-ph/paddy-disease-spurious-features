"""EDA for the Kaggle "Paddy Doctor: Paddy Disease Classification" data.

Expected layout under --data-dir (as downloaded from Kaggle):
    train.csv                 columns: image_id, label, variety, age
    train_images/<label>/<image_id>
    test_images/*.jpg         (no labels)

Usage:
    python scripts/eda.py --data-dir data/raw/paddy-disease-classification --out reports/eda

Questions this answers (write-up: reports/eda/eda_summary.md, data/README.md):
  * class distribution / imbalance ratio
  * variety and age distribution, and how label relates to them (shortcut risk)
  * which varieties could serve as a held-out domain (enough images, all classes)
  * image sizes, exact and near-duplicate images (leakage / label noise)
"""
import argparse
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.stats import chi2_contingency
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.data.dedup import hamming_pairs, hamming_pairs_between, hashes_to_u64, probe_images

SEED = 0
NEAR_DUP_MAX_DIST = 4  # of 64 bits (dHash 8x8)


# ----------------------------------------------------------------------------- loading
def load_train(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "train.csv")
    need = {"image_id", "label", "variety", "age"}
    if not need.issubset(df.columns):
        raise SystemExit(f"train.csv is missing columns: {sorted(need - set(df.columns))}")
    ids = df["image_id"].astype(str)
    ids = ids.where(ids.str.contains(r"\.\w+$"), ids + ".jpg")
    df["image_id"] = ids
    df["path"] = [data_dir / "train_images" / lab / i for lab, i in zip(df["label"], ids)]
    df["exists"] = [p.is_file() for p in df["path"]]
    return df


# ----------------------------------------------------------------------------- image probing
def probe_all(paths, desc, workers):
    paths = list(paths)
    return probe_images(paths, workers, progress=partial(tqdm, total=len(paths), desc=desc))


# ----------------------------------------------------------------------------- statistics
def cramers_v(a, b):
    """Bias-corrected Cramer's V between two categorical series."""
    ct = pd.crosstab(a, b)
    if min(ct.shape) < 2:
        return float("nan")
    chi2 = chi2_contingency(ct)[0]
    n = ct.values.sum()
    r, k = ct.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else float("nan")


def metadata_only_baseline(df, cols, n_splits=5):
    """5-fold CV of a lookup-table classifier: predict the majority label of the
    metadata group (fallback: global majority). Measures how much label information
    the metadata alone carries -- an upper-ish bound on what a shortcut could give."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    accs, f1s = [], []
    for tr, te in skf.split(df, df["label"]):
        trn, tst = df.iloc[tr], df.iloc[te]
        glob = trn["label"].value_counts().idxmax()
        if cols:
            table = (trn.groupby(cols)["label"]
                     .agg(lambda s: s.value_counts().idxmax()).rename("pred").reset_index())
            m = tst.merge(table, on=cols, how="left")
            pred = m["pred"].fillna(glob).values
        else:
            pred = np.full(len(tst), glob)
        accs.append(float((pred == tst["label"].values).mean()))
        f1s.append(float(f1_score(tst["label"].values, pred, average="macro", zero_division=0)))
    return {"acc": float(np.mean(accs)), "macro_f1": float(np.mean(f1s))}


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- figures
def fig_bar(series, title, path, horizontal=False, rot=45):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if horizontal:
        series.sort_values().plot.barh(ax=ax, color="#3b7ea1")
    else:
        series.plot.bar(ax=ax, color="#3b7ea1")
        plt.setp(ax.get_xticklabels(), rotation=rot, ha="right")
    ax.set_title(title)
    for c in ax.containers:
        ax.bar_label(c, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_heatmap(ct, title, path, fmt="d", normalize=False):
    data = ct.div(ct.sum(axis=1), axis=0) if normalize else ct
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(ct) + 2)))
    sns.heatmap(data, annot=True, fmt=".2f" if normalize else fmt, cmap="viridis",
                ax=ax, cbar=False, annot_kws={"size": 7})
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_samples(df, path, per_class=4, thumb=160):
    labels = sorted(df["label"].unique())
    fig, axes = plt.subplots(len(labels), per_class, figsize=(per_class * 2, len(labels) * 2))
    axes = np.atleast_2d(axes)
    rng = np.random.default_rng(SEED)
    for r, lab in enumerate(labels):
        sub = df[(df["label"] == lab) & df["exists"]]
        pick = sub.iloc[rng.permutation(len(sub))[:per_class]]
        for c in range(per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(pick):
                with Image.open(pick.iloc[c]["path"]) as im:
                    im = im.convert("RGB")
                    im.thumbnail((thumb, thumb))
                    ax.imshow(im)
            if c == 0:
                ax.set_title(lab, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/paddy-disease-classification"))
    ap.add_argument("--out", type=Path, default=Path("reports/eda"))
    ap.add_argument("--refresh", action="store_true", help="recompute cached image probes")
    ap.add_argument("--workers", type=int, default=2,
                    help="image-reading threads; 8 ran out of memory with ~1 GB free RAM")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    S = {}  # summary numbers -> summary.json
    md = ["# EDA summary", ""]

    df = load_train(args.data_dir)
    S["n_train_rows"] = int(len(df))
    S["n_missing_files"] = int((~df["exists"]).sum())
    S["n_train_ids_duplicated"] = int(df["image_id"].duplicated().sum())
    S["null_counts"] = {k: int(v) for k, v in df[["label", "variety", "age"]].isna().sum().items()}
    md += [f"- Train rows: **{S['n_train_rows']}**, missing files: **{S['n_missing_files']}**, "
           f"duplicate image_ids: **{S['n_train_ids_duplicated']}**, nulls: {S['null_counts']}"]
    if S["n_missing_files"]:
        print(f"WARNING: {S['n_missing_files']} files listed in train.csv were not found; "
              "they are excluded from image-based checks.")
    df = df.reset_index(drop=True)

    # ---- 1. class distribution --------------------------------------------------------
    cls = df["label"].value_counts()
    S["classes"] = {k: int(v) for k, v in cls.items()}
    S["n_classes"] = int(len(cls))
    S["imbalance_ratio_max_over_min"] = float(cls.max() / cls.min())
    cls_tab = pd.DataFrame({"count": cls, "share_%": (100 * cls / cls.sum()).round(2)}).reset_index()
    cls_tab.columns = ["label", "count", "share_%"]
    cls_tab.to_csv(out / "class_distribution.csv", index=False)
    fig_bar(cls, "Class distribution (train)", out / "class_distribution.png")
    md += ["", "## Class distribution",
           f"Imbalance ratio (max/min): **{S['imbalance_ratio_max_over_min']:.2f}**", "",
           md_table(cls_tab)]

    # ---- 2. variety & age ----------------------------------------------------------------
    var = df["variety"].value_counts()
    S["varieties"] = {k: int(v) for k, v in var.items()}
    S["largest_variety_share"] = float(var.iloc[0] / var.sum())
    fig_bar(var, "Variety distribution (train)", out / "variety_distribution.png")
    S["age"] = {k: float(v) for k, v in df["age"].describe().items()}
    S["age_n_unique"] = int(df["age"].nunique())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["age"].dropna(), bins=range(int(df["age"].min()), int(df["age"].max()) + 2),
            color="#3b7ea1")
    ax.set_title("Age (days) distribution")
    fig.tight_layout()
    fig.savefig(out / "age_distribution.png", dpi=150)
    plt.close(fig)
    md += ["", "## Variety / age",
           f"Largest variety share: **{100 * S['largest_variety_share']:.1f}%** "
           f"({var.index[0]}); {len(var)} varieties; age takes {S['age_n_unique']} distinct values "
           f"(min {S['age']['min']:.0f}, max {S['age']['max']:.0f}).", "",
           md_table(pd.DataFrame({"variety": var.index, "count": var.values,
                                  "share_%": (100 * var.values / var.sum()).round(2)}))]

    # ---- 3. label vs metadata (shortcut risk) ------------------------------------------
    ct_var = pd.crosstab(df["variety"], df["label"])
    ct_var.to_csv(out / "variety_x_label_counts.csv")
    fig_heatmap(ct_var, "Variety x label (counts)", out / "variety_x_label_counts.png")
    fig_heatmap(ct_var, "Variety x label (row-normalised: P(label | variety))",
                out / "variety_x_label_rownorm.png", normalize=True)
    age_bin = pd.cut(df["age"], bins=[0, 55, 60, 65, 70, 75, 200]).astype(str)
    ct_age = pd.crosstab(age_bin, df["label"])
    fig_heatmap(ct_age, "Age bin x label (counts)", out / "ageBin_x_label_counts.png")

    S["assoc"] = {
        "label_vs_variety": {"cramers_v": cramers_v(df["label"], df["variety"]),
                              "nmi": float(normalized_mutual_info_score(df["label"], df["variety"]))},
        "label_vs_age": {"cramers_v": cramers_v(df["label"], df["age"]),
                          "nmi": float(normalized_mutual_info_score(df["label"], df["age"]))},
    }
    S["metadata_only_baseline_5fold"] = {
        "majority_class": metadata_only_baseline(df, []),
        "variety": metadata_only_baseline(df, ["variety"]),
        "age": metadata_only_baseline(df, ["age"]),
        "variety+age": metadata_only_baseline(df, ["variety", "age"]),
    }
    mb = pd.DataFrame([{"predictor": k, **v} for k, v in S["metadata_only_baseline_5fold"].items()])
    md += ["", "## Is the label predictable from metadata alone?",
           "Association of label with metadata (Cramer's V is bias-corrected; NMI in [0,1]):", "",
           md_table(pd.DataFrame([{"pair": k, **v} for k, v in S["assoc"].items()])),
           "", "5-fold CV of a metadata-only lookup classifier (no image used). "
           "Compare against the majority-class row; a large gap means label correlates with "
           "variety/age, i.e. a shortcut exists.", "", md_table(mb)]

    # ---- 4. leave-variety-out feasibility ---------------------------------------------
    rows = []
    for v, g in df.groupby("variety"):
        per = g["label"].value_counts()
        rows.append({"variety": v, "n_images": len(g), "n_classes_present": int(per.size),
                     "min_class_count": int(per.min()), "missing_classes":
                     ", ".join(sorted(set(cls.index) - set(per.index))) or "-"})
    lvo = pd.DataFrame(rows).sort_values("n_images", ascending=False)
    lvo.to_csv(out / "leave_variety_out_feasibility.csv", index=False)
    md += ["", "## Held-out-variety feasibility",
           "A variety is a usable held-out domain if it has enough images and covers (nearly) "
           "all classes.", "", md_table(lvo)]

    # ---- 5. image probing: sizes, exact + near duplicates --------------------------------
    df_ok = df[df["exists"]].reset_index(drop=True)
    cache = out / "image_probe_train.csv"
    if cache.exists() and not args.refresh and len(pd.read_csv(cache)) == len(df_ok):
        pr = pd.read_csv(cache, dtype={"dhash": str})
    else:
        pr = probe_all(list(df_ok["path"]), "probe train", args.workers)
        pr.insert(0, "image_id", df_ok["image_id"].values)
        pr.to_csv(cache, index=False)
    df_ok = pd.concat([df_ok, pr[["w", "h", "md5", "dhash"]]], axis=1)

    size_counts = (df_ok["w"].astype(str) + "x" + df_ok["h"].astype(str)).value_counts()
    S["image_sizes_top"] = {k: int(v) for k, v in size_counts.head(8).items()}
    S["n_distinct_image_sizes"] = int(size_counts.size)
    md += ["", "## Image sizes (WxH)", md_table(pd.DataFrame(
        {"size": size_counts.head(8).index, "count": size_counts.head(8).values})),
        f"({S['n_distinct_image_sizes']} distinct sizes in total)"]

    exact = df_ok[df_ok.duplicated("md5", keep=False)].sort_values("md5")
    exact.drop(columns=["path", "exists"]).to_csv(out / "exact_duplicates.csv", index=False)
    S["exact_duplicate_images"] = int(len(exact))
    S["exact_duplicate_groups"] = int(exact["md5"].nunique())
    S["exact_dup_groups_with_conflicting_labels"] = int(
        (exact.groupby("md5")["label"].nunique() > 1).sum())

    hashes = hashes_to_u64(df_ok["dhash"])
    pdf = hamming_pairs(hashes, NEAR_DUP_MAX_DIST)
    if len(pdf):
        a, b = df_ok.loc[pdf["i"]].reset_index(drop=True), df_ok.loc[pdf["j"]].reset_index(drop=True)
        pdf["id_a"], pdf["id_b"] = a["image_id"], b["image_id"]
        pdf["label_a"], pdf["label_b"] = a["label"], b["label"]
        pdf["variety_a"], pdf["variety_b"] = a["variety"], b["variety"]
        pdf["age_a"], pdf["age_b"] = a["age"], b["age"]
        pdf["same_label"] = pdf["label_a"] == pdf["label_b"]
        pdf["same_variety"] = pdf["variety_a"] == pdf["variety_b"]
        pdf["same_age"] = pdf["age_a"] == pdf["age_b"]
    pdf.drop(columns=["i", "j"]).to_csv(out / "near_duplicate_pairs.csv", index=False)
    S["near_duplicate_pairs_dist<=%d" % NEAR_DUP_MAX_DIST] = int(len(pdf))
    if len(pdf):
        S["near_dup_pairs_cross_label"] = int((~pdf["same_label"]).sum())
        S["near_dup_pairs_same_variety_frac"] = float(pdf["same_variety"].mean())
        S["near_dup_pairs_same_age_frac"] = float(pdf["same_age"].mean())
        for t in (0, 2, 4):
            S[f"near_dup_pairs_dist<={t}"] = int((pdf["dist"] <= t).sum())
    n_involved = len(set(pdf["id_a"]) | set(pdf["id_b"])) if len(pdf) else 0
    S["images_in_any_near_dup_pair"] = int(n_involved)
    md += ["", "## Duplicates",
           f"- Exact (md5) duplicate images: **{S['exact_duplicate_images']}** in "
           f"{S['exact_duplicate_groups']} groups "
           f"({S['exact_dup_groups_with_conflicting_labels']} groups with conflicting labels)",
           f"- Near-duplicate pairs (dHash distance <= {NEAR_DUP_MAX_DIST}/64): "
           f"**{len(pdf)}** pairs, {n_involved} images involved"]
    if len(pdf):
        md += [f"  - cross-label pairs (possible label noise): {S['near_dup_pairs_cross_label']}",
               f"  - same variety: {100 * S['near_dup_pairs_same_variety_frac']:.1f}%, "
               f"same age: {100 * S['near_dup_pairs_same_age_frac']:.1f}% "
               "(high values => random splits would leak; group by variety/age)"]
    md += ["- dHash is a coarse 64-bit hash: inspect `near_duplicate_pairs.csv` visually before "
           "dropping anything."]

    # ---- 6. test images ------------------------------------------------------------------
    test_files = sorted((args.data_dir / "test_images").glob("*.*"))
    S["n_test_images"] = len(test_files)
    if test_files:
        tp = probe_all(test_files, "probe test", args.workers)
        S["test_distinct_sizes"] = int((tp["w"].astype(str) + "x" + tp["h"].astype(str)).nunique())
        tr_md5 = set(df_ok["md5"])
        S["test_exact_dups_of_train"] = int(tp["md5"].isin(tr_md5).sum())
        tn = hamming_pairs_between(hashes_to_u64(tp["dhash"]), hashes, NEAR_DUP_MAX_DIST)
        S["test_images_with_near_dup_in_train"] = int(tn["i"].nunique())
        md += ["", "## Test images (unlabelled)",
               f"- {S['n_test_images']} images, {S['test_distinct_sizes']} distinct sizes",
               f"- exact duplicates of a train image: {S['test_exact_dups_of_train']}",
               f"- with a near-duplicate in train (dist <= {NEAR_DUP_MAX_DIST}): "
               f"{S['test_images_with_near_dup_in_train']}"]

    # ---- 7. sample grid + write ---------------------------------------------------------
    fig_samples(df, out / "samples_per_class.png")
    md += ["", "Figures and CSVs are in this folder; raw numbers in `summary.json`."]
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=float), encoding="utf-8")
    (out / "eda_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Done. See {out / 'eda_summary.md'}")


if __name__ == "__main__":
    main()
