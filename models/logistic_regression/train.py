"""Train the logistic regression on the Breast Cancer Wisconsin dataset.

Usage:
    uv run python -m models.logistic_regression.train
    uv run python -m models.logistic_regression.train --output /tmp/logistic_regression
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from framework.evaluation import evaluate
from framework.validation import validate_model
from models.logistic_regression.model import build_model

RANDOM_STATE = 42
VALIDATION_ROWS = 100


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load the bundled breast cancer dataset as features and labels."""
    # Bundled with scikit-learn: no download needed.
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target.to_numpy()


def main() -> None:
    """Train, evaluate, and optionally save the classifier from CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Directory to save the trained model")
    args = parser.parse_args()

    features, target = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=RANDOM_STATE
    )

    model = build_model(numeric_features=list(features.columns))
    # Fail fast on a small sample before spending time on the real training.
    validate_model(model, x_train.iloc[:VALIDATION_ROWS], y_train[:VALIDATION_ROWS], trainable=True)
    model.fit(x_train, y_train)

    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
    }
    for name, value in evaluate(model, x_test, y_test, metrics).items():
        print(f"{name:<10} {value:.4f}")

    if args.output:
        model.save(args.output)
        print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
