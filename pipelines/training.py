"""Train any model in `models/` with ZenML.

Usage:
    uv run --group mlops python -m pipelines.training --model xgboost_classifier
"""

import argparse

from zenml import pipeline

from models import available_models
from steps.evaluate import evaluate
from steps.load_data import load_data
from steps.split_data import split_data
from steps.train import train


@pipeline
def training_pipeline(model_name: str) -> None:
    """Load the data, split it, train the chosen model and evaluate it."""
    features, target = load_data()
    x_train, x_test, y_train, y_test = split_data(features, target)
    model = train(model_name, x_train, y_train)
    evaluate(model, x_test, y_test)


def run_training(model_name: str) -> dict[str, float]:
    """Run the training pipeline and return the metrics of its evaluate step."""
    run = training_pipeline(model_name=model_name)
    if run is None:
        # Only happens when an orchestrator schedules the run instead of running it now.
        raise RuntimeError("The training pipeline was not run: there are no metrics to report")
    metrics: dict[str, float] = run.steps["evaluate"].outputs["metrics"][0].load()
    return metrics


def main() -> None:
    """Run the training pipeline for the model named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=available_models())
    args = parser.parse_args()

    for name, value in run_training(args.model).items():
        print(f"{name:<10} {value:.4f}")


if __name__ == "__main__":
    main()
