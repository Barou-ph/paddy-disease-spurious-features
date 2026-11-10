# Paddy Disease Classification

## Research Topic

Investigating and Mitigating Spurious Visual Features in Deep Learning for Paddy Disease Classification

## Research Question

Does a deep learning model actually learn disease-specific visual features, or does it rely on spurious characteristics such as background, leaf appearance, variety, and crop age?

## Dataset

Paddy Doctor Dataset

- 10 disease classes
- 10,407 labeled training images
- Additional metadata including variety and age

## Baseline

Pretrained EfficientNet-B0 with Cross-Entropy Loss.

## Research Direction

We investigate whether the model relies on spurious visual features and evaluate whether a proposed training/modeling strategy can reduce such reliance.

## Project Structure

```text
configs/       Experiment configurations
data/          Dataset documentation and local data
src/           Source code
experiments/   Experiment outputs
notebooks/     Exploration and analysis
reports/      Report materials
checkpoints/   Local model checkpoints
```
