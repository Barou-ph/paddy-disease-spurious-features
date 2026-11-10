# Shared Core

This is the common foundation for the Paddy Disease project.

## Setup

Activate the project virtual environment, then:

```bash
pip install -r requirements.txt
```

## Smoke test

From the repository root:

```bash
python scripts/smoke_test.py
```

Expected result includes:

```text
Shared core smoke test: OK
```

## Important

Do not duplicate the training loop or metric definitions in feature branches.

Use:

- `src/core/` for shared infrastructure
- `src/data/` for the common dataset contract
- `src/models/factory.py` for model construction
- `src/training/` for the common training loop
- `src/evaluation/` for shared metrics
- `configs/` for experiment settings

Research-specific code belongs outside the shared infrastructure.
