# Lead Experiment Results

## 1. Experiment Configuration

| Setting | Value |
|---|---|
| Model | EfficientNet-B0 (pretrained) |
| Loss | Cross-Entropy |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | Cosine |
| Input size | 224x224 |
| Seed | 42 |
| Primary metric | Macro-F1 |

## 2. Random Split Baseline

The baseline was trained and evaluated using the random split.

| Metric | Result |
|---|---:|
| Accuracy | 0.9269 |
| Macro Precision | 0.9152 |
| Macro Recall | 0.9106 |
| Macro-F1 | 0.9118 |
| Best validation Macro-F1 | 0.9315 |
| Best epoch | 21 |

## 3. Split-specific Baseline Experiments

Separate EfficientNet-B0 models were trained for the variety, age_old, and age_young splits. Each model was evaluated on its corresponding in-domain test set.

| Split | Train samples | Validation samples | Test samples | Best epoch | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| variety | 4798 | 1033 | 1035 | 25 | 0.9633 | 0.9548 |
| age_old | 5974 | 1288 | 1281 | 16 | 0.9719 | 0.9680 |
| age_young | 6004 | 1287 | 1297 | 18 | 0.9668 | 0.9618 |

## 4. Out-of-Distribution Evaluation

Each split-specific checkpoint was evaluated on its corresponding held-out OOD test set.

| Training split | OOD condition | OOD samples | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| variety | Unseen varieties | 3347 | 0.4111 | 0.3308 |
| age_old | Unseen older-age group | 1708 | 0.4649 | 0.3206 |
| age_young | Unseen younger-age group | 1660 | 0.4205 | 0.3929 |

Macro-F1 is calculated over the classes present in the ground-truth labels of each evaluation subset.

## 5. Generalization Gap

The difference between in-domain test Macro-F1 and corresponding OOD Macro-F1 is shown below.

| Split | In-domain Macro-F1 | OOD Macro-F1 | Difference |
|---|---:|---:|---:|
| variety | 0.9548 | 0.3308 | 0.6240 |
| age_old | 0.9680 | 0.3206 | 0.6474 |
| age_young | 0.9618 | 0.3929 | 0.5689 |

The baseline achieves strong in-domain performance but experiences substantial performance degradation under the controlled distribution shifts.

## 6. Preliminary Interpretation

The results reveal a substantial generalization gap between in-domain and out-of-distribution evaluation.

The variety split shows that a model trained on ADT45 performs poorly on samples from unseen varieties. Similarly, the age-based experiments show substantial degradation when evaluating on age groups excluded from training.

These findings are consistent with the hypothesis that standard empirical risk minimization may learn features that do not generalize reliably across changes in data distribution.

However, these results alone do not identify which visual features cause the degradation. Further analysis using metadata-aligned subsets, Grad-CAM, controlled interventions, and the proposed method is required.

## 7. Next Experiments

1. Integrate and train the proposed method.
2. Evaluate the proposed method using the same split definitions and evaluation protocol.
3. Compare baseline and proposed method on both in-domain and OOD test sets.
4. Perform ablation experiments.
5. Analyze model attention using Grad-CAM.
6. Investigate class-wise errors and metadata-aligned versus counter-aligned samples.

## 8. Reproducibility

Baseline checkpoints:

- `experiments/runs/baseline_clean_random/best.pt`
- `experiments/runs/baseline_variety/best.pt`
- `experiments/runs/baseline_age_old/best.pt`
- `experiments/runs/baseline_age_young/best.pt`

Robustness evaluation outputs:

- `experiments/runs/robustness_variety/`
- `experiments/runs/robustness_age_old/`
- `experiments/runs/robustness_age_young/`

The random split evaluation is stored separately from the three split-specific OOD experiments.