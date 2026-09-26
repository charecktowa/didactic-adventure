# didactic-adventure

A small MLOps framework, built step by step: any model in `models/` is validated against the
framework's contracts, then trained and evaluated by a ZenML pipeline.

## Layout

| Path | What it holds |
|---|---|
| `src/framework/` | Contracts, connectors, validation and evaluation. Independent of any orchestrator. |
| `models/<name>/` | One model per package: `model.py` exposes `build_model()`. |
| `steps/`, `pipelines/` | ZenML steps and the training pipeline that uses the framework. |

## Run it locally

```bash
uv sync --group mlops
uv run --group mlops python -m pipelines.training --model logistic_regression
uv run pytest
```

## Run it with Docker

Needs Docker 23 or newer, where BuildKit is the default builder.

The image gives everyone the same Linux, Python and dependencies, whatever the host system.

```bash
docker compose build
docker compose run --rm app                      # trains logistic_regression
docker compose run --rm app python -m pipelines.training --model xgboost_classifier
docker compose run --rm app pytest
```

To browse the runs in ZenML's dashboard:

```bash
docker compose up -d dashboard                   # then open http://localhost:8237 (user: default, no password)
docker compose down                              # stop it; the runs stay in the volume
```

The source folder is mounted into the container, so code changes need no rebuild; rebuild after
changing dependencies. ZenML's runs are kept in the `zenml` volume between containers.
