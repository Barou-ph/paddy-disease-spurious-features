# Paddy Disease Classification

## Research Topic

**Investigating and Mitigating Spurious Visual Features in Deep Learning for
Paddy Disease Classification**

## Research Question

Does a deep learning model actually learn disease-specific visual features,
or does it rely on spurious characteristics such as background, leaf
appearance, variety, and crop age?

## Working Hypothesis

A standard classifier relies partly on non-disease-specific correlations.
Reducing this reliance should improve generalization to controlled or unseen
conditions while preserving competitive in-distribution macro-F1.

## Dataset

Paddy Doctor Dataset.

The current project plan targets:

- 10 disease classes
- 10,407 labeled training images
- metadata including variety and age
- an additional unlabeled test set

Raw data is not stored in Git.

## Baseline

Pretrained EfficientNet-B0 with Cross-Entropy loss.

Primary metric: **macro-F1**.

Supporting metrics:

- accuracy
- macro precision
- macro recall
- confusion matrix

## Quick start

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/smoke_test.py
```

The smoke test validates imports, configuration, model construction, and
environment detection without requiring the dataset.

## Baseline training

Prepare a manifest with:

```text
image_path,label
```

Then:

```bash
python scripts/train_baseline.py \
  --manifest data/processed/manifest.csv
```

The script creates a stratified train/validation/test split. For the final
project, use the explicit split manifests produced by the dataset/robustness
workflow when those become available.

## Baseline outputs

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

Generated outputs and checkpoints are ignored by Git.

## Project architecture

See [ARCHITECTURE.md](ARCHITECTURE.md).

## Team workflow

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Lead experiment plan

See [docs/lead_experiment_plan.md](docs/lead_experiment_plan.md).
