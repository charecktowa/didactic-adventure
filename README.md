# didactic-adventure

A small ML framework to learn MLOps step by step.

## Layout

- `src/framework/contracts/model.py`: the `Model` contract (`predict`, `save`, `load`) and the
  optional `Trainable` contract (`fit`). They are `Protocol`s: a model fulfils them by shape, it
  does not need to inherit from anything.
- `src/framework/evaluation.py`: `evaluate()` works with any `Model`.
- `models/`: concrete models. `xgboost_classifier` is an XGBoost classifier inside a
  scikit-learn `Pipeline`.

## Train a model locally

```bash
uv sync --extra xgboost
uv run python -m models.xgboost_classifier.train
# optionally save it
uv run python -m models.xgboost_classifier.train --output /tmp/xgboost_classifier
```

It trains on the Breast Cancer Wisconsin dataset, bundled with scikit-learn, and prints the test
metrics.

## Checks

Model dependencies are optional extras, so install them all before running the checks:

```bash
uv sync --all-extras
uv run ruff check . && uv run ruff format --check . && uv run mypy . && uv run pytest
```
