"""Train the XGBoost classifier on the Breast Cancer Wisconsin dataset.

Usage:
    uv run python -m models.xgboost_classifier.train
    uv run python -m models.xgboost_classifier.train --output /tmp/xgboost_classifier
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from framework.evaluation import evaluate
from models.xgboost_classifier.model import XGBoostClassifier

RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    # Bundled with scikit-learn: no download needed.
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target.to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Directory to save the trained model")
    args = parser.parse_args()

    features, target = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=RANDOM_STATE
    )

    model = XGBoostClassifier(
        numeric_features=list(features.columns),
        xgb_params={
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": RANDOM_STATE,
        },
    ).fit(x_train, y_train)

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
