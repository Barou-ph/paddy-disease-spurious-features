# EDA summary

- Train rows: **10407**, missing files: **0**, duplicate image_ids: **0**, nulls: {'label': 0, 'variety': 0, 'age': 0}

## Class distribution
Imbalance ratio (max/min): **5.23**

| label | count | share_% |
|---|---|---|
| normal | 1764 | 16.950 |
| blast | 1738 | 16.700 |
| hispa | 1594 | 15.320 |
| dead_heart | 1442 | 13.860 |
| tungro | 1088 | 10.450 |
| brown_spot | 965 | 9.270 |
| downy_mildew | 620 | 5.960 |
| bacterial_leaf_blight | 479 | 4.600 |
| bacterial_leaf_streak | 380 | 3.650 |
| bacterial_panicle_blight | 337 | 3.240 |

## Variety / age
Largest variety share: **67.2%** (ADT45); 10 varieties; age takes 18 distinct values (min 45, max 82).

| variety | count | share_% |
|---|---|---|
| ADT45 | 6992 | 67.190 |
| KarnatakaPonni | 988 | 9.490 |
| Ponni | 657 | 6.310 |
| AtchayaPonni | 461 | 4.430 |
| Zonal | 399 | 3.830 |
| AndraPonni | 377 | 3.620 |
| Onthanel | 351 | 3.370 |
| IR20 | 114 | 1.100 |
| RR | 36 | 0.350 |
| Surya | 32 | 0.310 |

## Is the label predictable from metadata alone?
Association of label with metadata (Cramer's V is bias-corrected; NMI in [0,1]):

| pair | cramers_v | nmi |
|---|---|---|
| label_vs_variety | 0.273 | 0.161 |
| label_vs_age | 0.336 | 0.216 |

5-fold CV of a metadata-only lookup classifier (no image used). Compare against the majority-class row; a large gap means label correlates with variety/age, i.e. a shortcut exists.

| predictor | acc | macro_f1 |
|---|---|---|
| majority_class | 0.170 | 0.029 |
| variety | 0.306 | 0.189 |
| age | 0.365 | 0.215 |
| variety+age | 0.441 | 0.352 |

## Held-out-variety feasibility
A variety is a usable held-out domain if it has enough images and covers (nearly) all classes.

| variety | n_images | n_classes_present | min_class_count | missing_classes |
|---|---|---|---|---|
| ADT45 | 6992 | 10 | 107 | - |
| KarnatakaPonni | 988 | 7 | 10 | bacterial_panicle_blight, normal, tungro |
| Ponni | 657 | 9 | 6 | bacterial_leaf_streak |
| AtchayaPonni | 461 | 4 | 27 | bacterial_leaf_blight, bacterial_leaf_streak, bacterial_panicle_blight, brown_spot, dead_heart, normal |
| Zonal | 399 | 8 | 6 | bacterial_panicle_blight, normal |
| AndraPonni | 377 | 4 | 14 | bacterial_leaf_blight, bacterial_leaf_streak, brown_spot, downy_mildew, normal, tungro |
| Onthanel | 351 | 8 | 4 | bacterial_panicle_blight, brown_spot |
| IR20 | 114 | 3 | 10 | bacterial_leaf_streak, bacterial_panicle_blight, brown_spot, downy_mildew, hispa, normal, tungro |
| RR | 36 | 2 | 18 | bacterial_leaf_blight, bacterial_leaf_streak, bacterial_panicle_blight, blast, brown_spot, downy_mildew, hispa, tungro |
| Surya | 32 | 4 | 2 | bacterial_leaf_streak, bacterial_panicle_blight, brown_spot, dead_heart, hispa, normal |

## Image sizes (WxH)
| size | count |
|---|---|
| 480x640 | 10403 |
| 640x480 | 4 |
(2 distinct sizes in total)

## Duplicates
- Exact (md5) duplicate images: **146** in 72 groups (0 groups with conflicting labels)
- Near-duplicate pairs (dHash distance <= 4/64): **2268** pairs, 2451 images involved
  - cross-label pairs (possible label noise): 86
  - same variety: 98.0%, same age: 99.8% (high values => random splits would leak; group by variety/age)
- dHash is a coarse 64-bit hash: inspect `near_duplicate_pairs.csv` visually before dropping anything.

## Test images (unlabelled)
- 3469 images, 2 distinct sizes
- exact duplicates of a train image: 51
- with a near-duplicate in train (dist <= 4): 914

Figures and CSVs are in this folder; raw numbers in `summary.json`.
