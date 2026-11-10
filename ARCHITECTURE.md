# Project Architecture

## Goal

This repository is shared by three team members. The shared code defines the
common contract for data loading, reproducibility, model construction,
training, evaluation, and experiment configuration.

Research-specific work should be added on top of this foundation rather than
copying and modifying the whole training pipeline.

## Data flow

```text
YAML config
    |
    +--> seed / device / logger
    |
    +--> dataset -> transforms -> dataloader
    |
    +--> model factory -> model
    |
    +--> trainer -> checkpoint
    |
    +--> evaluator -> metrics -> results
```

## Shared modules

### `src/core/`

Stable project infrastructure.

- `config.py`: YAML loading and nested config access.
- `seed.py`: reproducible random seeds.
- `device.py`: CPU/CUDA selection.
- `logger.py`: consistent experiment logging.
- `checkpoint.py`: checkpoint save/load.
- `reproducibility.py`: environment metadata.

### `src/data/`

Common dataset interface and preprocessing.

The manifest contract is:

```text
image_path,label
```

Optional metadata columns can include:

```text
variety,age,group_id,split
```

Do not fit data-dependent preprocessing on validation/test data.

### `src/models/`

`factory.py` is the common entry point for model creation.

The current baseline is pretrained EfficientNet-B0 with a 10-class classifier.

New research models can be added to the factory without rewriting the trainer.

### `src/training/`

Common optimization loop, loss, and scheduler.

Research-specific training strategies should expose compatible model/loss/hooks
rather than copy the complete loop.

### `src/evaluation/`

All experiments should use the same metric definitions.

Primary metric:

```text
macro-F1
```

Supporting metrics:

```text
accuracy
macro precision
macro recall
confusion matrix
```

## Experiment configuration

Put experiment choices in `configs/*.yaml`.

Do not hard-code learning rate, batch size, epochs, image size, seed, or model
settings inside training scripts.

Example:

```yaml
seed: 42

data:
  image_size: 224
  batch_size: 32
  num_workers: 4

model:
  name: efficientnet_b0
  pretrained: true
  num_classes: 10

training:
  epochs: 30
  learning_rate: 0.0001
  weight_decay: 0.0001
  loss: cross_entropy
  scheduler: none

evaluation:
  primary_metric: f1_macro

output:
  checkpoint_dir: checkpoints/baseline
```

## Team ownership

Suggested ownership:

```text
Lead:
    core/
    models/
    training/

Person 2:
    data/
    robustness evaluation

Person 3:
    research method / analysis / Grad-CAM / ablation
```

Ownership does not prevent review. A shared-core change should be discussed
before merging because it can affect every experiment.

## Research isolation

The following are NOT core infrastructure:

- Grad-CAM
- spurious-feature mitigation
- proposed architecture
- special losses
- robustness subsets
- ablation-specific logic

These should live in research-specific modules/configs so that the baseline
remains reproducible and the contribution can be ablated cleanly.

## Reproducibility contract

Every reported experiment should record:

- config
- seed
- model/pretrained weight choice
- data split
- preprocessing
- optimizer
- learning rate
- scheduler
- epochs
- best checkpoint
- test metrics
- environment/hardware information
