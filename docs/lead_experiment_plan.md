# Lead Experiment Plan

## 1. Research Question

> Does a deep learning model actually learn disease-specific visual features,
> or does it rely on spurious characteristics such as background, leaf
> appearance, variety, and crop age?

## 2. Working Hypothesis

**H1:** A standard Empirical Risk Minimization classifier trained on Paddy
Doctor images relies partly on correlations that are not disease-specific,
including background, leaf appearance, variety, and crop age. Reducing this
reliance should improve generalization on controlled or unseen conditions
while preserving competitive in-distribution macro-F1.

This hypothesis is tested rather than assumed. Grad-CAM and controlled
subgroup evaluation are evidence tools, not the contribution by themselves.

## 3. Baseline

- Model: pretrained EfficientNet-B0.
- Number of classes: 10.
- Loss: Cross-Entropy.
- Optimizer: AdamW.
- Primary metric: macro-F1.
- Supporting metrics: accuracy, macro precision, macro recall, confusion matrix.
- Input size: 224 x 224.
- Seed: 42 for the initial baseline.
- Validation is used for model selection/early stopping.
- Test data remains reserved for final evaluation.

## 4. Experiment Matrix

| ID | Experiment | Purpose |
|---|---|---|
| EXP-01 | EfficientNet-B0 baseline | Establish reference performance |
| EXP-02 | Baseline subgroup evaluation | Measure variety/age/generalization gaps |
| EXP-03 | Controlled shortcut evaluation | Test sensitivity to background/leaf appearance |
| EXP-04 | Proposed method | Test the research contribution |
| EXP-05 | Ablation A | Remove one component of proposed method |
| EXP-06 | Ablation B | Remove another component |
| EXP-07 | Final robustness comparison | Compare baseline vs proposed under predefined difficult subsets |

The exact perturbations/subgroups for EXP-03 and EXP-07 must be defined before
final results are inspected.

## 5. Fair Comparison Rules

Baseline and proposed method should use the same:

- test set
- primary metric
- label mapping
- image resolution unless the method requires otherwise
- evaluation protocol
- reporting convention

Changes to data, preprocessing, optimizer, or training budget must be
reported explicitly.

## 6. Reproducibility

Each experiment records:

- configuration
- seed
- model/pretrained weights
- split
- preprocessing
- optimizer
- learning rate
- scheduler
- epoch count
- best checkpoint
- test metrics
- environment/hardware

The initial baseline uses seed 42. Key final comparisons should use multiple
seeds when compute permits.

## 7. Expected Baseline Outputs

A completed baseline run should produce:

```text
experiments/runs/baseline/
├── best.pt
├── last.pt
├── config.json
├── environment.json
├── label_mapping.json
├── history.csv
├── test_metrics.json
├── confusion_matrix.csv
├── summary.json
└── train.log
```

Do not commit checkpoints or generated experiment outputs to Git.
