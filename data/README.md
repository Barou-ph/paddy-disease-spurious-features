# Data

## Source

Paddy Doctor, distributed as the Kaggle competition
[Paddy Doctor: Paddy Disease Classification](https://www.kaggle.com/competitions/paddy-disease-classification)
(Petchiammal et al., *Paddy Doctor: A Visual Image Dataset for Automated Paddy
Disease Classification and Benchmarking*, CODS-COMAD 2023). Check the
competition's data rules before redistributing images; this repository only
stores derived metadata (image ids, labels, splits), never the images.

- 10,407 labelled field photos, 480x640 (4 are 640x480), 10 classes
  (9 diseases + `normal`)
- metadata per image: `variety` (10 rice varieties) and `age` (crop age in days, 45-82)
- 3,469 unlabelled `test_images/` (Kaggle leaderboard set). We do **not** use them:
  without labels they cannot be evaluated.

## Setup

1. Download the competition zip, e.g. with the Kaggle CLI:

   ```bash
   kaggle competitions download -c paddy-disease-classification -p data/raw
   ```

2. Extract it so the layout is:

   ```text
   data/raw/paddy-disease-classification/
       train.csv                       image_id, label, variety, age
       train_images/<label>/<image_id>
       test_images/
       sample_submission.csv
   ```

3. The split CSVs in `data/splits/` are committed, so nothing else is needed to
   train. To regenerate them (a few minutes the first time, while hashing images):

   ```bash
   python scripts/prepare_splits.py
   ```

   Re-running with the same arguments reproduces the committed files byte for byte.

## Audit findings

Full numbers and figures: `reports/eda/eda_summary.md` (from `python scripts/eda.py`).

| Finding | Number | Consequence |
|---|---|---|
| Class imbalance | `normal` 1,764 vs `bacterial_panicle_blight` 337 (5.2x) | macro-F1 is the primary metric |
| Variety skew | ADT45 = 67.2% of images; only variety with all 10 classes | ADT45 is the in-domain set for the `variety` split |
| Label depends on metadata | (variety, age) lookup without the image: 44.1% acc / 0.35 macro-F1, vs 17.0% / 0.03 for the majority class | metadata is a real shortcut; `meta_aligned` column marks it |
| Exact duplicates | 146 images in 72 groups, no label conflict | 74 extra copies dropped |
| Same photo, two labels | 86 near-identical cross-label pairs (dHash <= 4/64; a sample was checked by eye and they are the same photo), e.g. blast/downy_mildew, hispa/normal | 82 images dropped as label noise |
| Re-shots of one scene | near-duplicate pairs are 98% same variety and 99.8% same age (random pair: 47% / 15%), still true up to dHash 10/64 | images are grouped (dHash <= 8/64) and a group never spans two subsets |

## Splits

Every split is stratified by label and grouped by `group_id`; the in-domain part
is 70/15/15 train/val/test. Splits with a held-out domain add `test_ood.csv`.
Out-of-domain images that are near copies of in-domain images are removed from
`test_ood` (38 for `variety`, 3 for `age_young`).

| Split | In-domain (train/val/test) | `test_ood` | Use it to test |
|---|---|---|---|
| `random` | all clean images: 7,168 / 1,538 / 1,545 | - | in-distribution baseline |
| `variety` | ADT45: 4,798 / 1,033 / 1,035 | 3,347 images from the 9 other varieties, all 10 classes | reliance on variety-specific appearance |
| `age_old` | age <= 70: 5,974 / 1,288 / 1,281 | 1,708 images age >= 72; **no `downy_mildew`** | generalisation to later growth stages |
| `age_young` | age >= 55: 6,004 / 1,287 / 1,297 | 1,660 images age <= 50; **no `bacterial_panicle_blight`** | generalisation to earlier growth stages |

Per-subset class counts: `data/splits/summary.md`.

### Columns

`data/splits/<split>/<subset>.csv`:

| Column | Meaning |
|---|---|
| `image_id`, `label`, `variety`, `age` | from `train.csv` |
| `image_path` | `<label>/<image_id>`, relative to `data.image_root` |
| `group_id` | near-duplicate group; shared by re-shots of the same scene |
| `split` | `train`, `val`, `test` or `test_ood` |
| `meta_pred` | label predicted from (variety, age) alone, by a lookup fitted on this split's train subset |
| `meta_aligned` | `meta_pred == label`. `False` = shortcut-conflicting sample |

`data/manifest.csv` lists all 10,407 images with `group_id`, `kept` and
`drop_reason` (`label_conflict` / `exact_duplicate`).

### Loading

```python
from src.core.config import load_config
from src.data.paddy import build_split_datasets

config = load_config("configs/baseline.yaml")   # data.split: random
datasets = build_split_datasets(config)          # {"train", "val", "test"[, "test_ood"]}
```

Each dataset's `.df` keeps all columns, and `batch["index"]` indexes into it, so
per-sample results can be joined back to `variety`, `age` and `meta_aligned`.

### Rules for experiments

- Always use `src.data.paddy.LABEL_TO_INDEX` (done by `build_split_datasets`).
  `age_old/test_ood` and `age_young/test_ood` miss a class, so a mapping inferred
  from the labels present would shift class indices.
- On a `test_ood` that misses a class, report macro-F1 over the classes present
  in that subset (and say so); otherwise the absent class adds a zero F1 whenever
  the model predicts it.
- Choose checkpoints and hyper-parameters on `val` only. `test` and `test_ood`
  are for final numbers.
- Shortcut reliance: compare macro-F1 on `meta_aligned == True` vs `False` rows of
  the same test subset. A large gap means the model leans on variety/age cues.
