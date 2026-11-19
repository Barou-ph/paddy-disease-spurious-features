# Lead Experiment Results

## Baseline

Model: EfficientNet-B0 pretrained
Loss: Cross-Entropy
Optimizer: AdamW
Learning rate: 1e-4
Weight decay: 1e-4
Scheduler: Cosine
Input size: 224x224
Seed: 42
Primary metric: Macro-F1

## Random split

| Metric | Result |
|---|---:|
| Accuracy | 0.9269 |
| Macro Precision | 0.9152 |
| Macro Recall | 0.9106 |
| Macro-F1 | 0.9118 |
| Best validation Macro-F1 | 0.9315 |
| Best epoch | 21 |

## Robustness evaluation

| Split | Evaluation | Samples | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| random | test | 1545 | 0.9269 | 0.9118 |
| variety | test | 1035 | 0.9903 | 0.9893 |
| variety | OOD | 3347 | 0.9713 | 0.9540 |
| age_old | test | 1281 | 0.9703 | 0.9685 |
| age_old | OOD | 1708 | 0.9895 | 0.9800 |
| age_young | test | 1297 | 0.9807 | 0.9759 |
| age_young | OOD | 1660 | 0.9789 | 0.9759 |

## Current interpretation

The EfficientNet-B0 baseline achieves 91.18% test Macro-F1 on the random split.

The controlled variety and age evaluations also show high performance, including 95.40% Macro-F1 on the unseen-variety OOD set. Therefore, these results alone do not establish that the baseline strongly depends on spurious visual features.

The next experiment should directly test shortcut dependence using controlled interventions, metadata-aligned subsets, or the proposed method rather than assuming poor OOD performance.
