from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from framework.contracts.model import Model
from framework.evaluation import evaluate


class ConstantModel:
    """Test double: fulfils the Model contract by shape, without inheriting from it."""

    def __init__(self, label: int) -> None:
        self.label = label

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.label)

    def save(self, directory: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path) -> Self:
        raise NotImplementedError


def test_constant_model_fulfils_contract() -> None:
    model: Model = ConstantModel(label=1)  # checked statically by mypy

    assert isinstance(model, Model)


def test_evaluate_returns_one_float_score_per_metric() -> None:
    features = pd.DataFrame({"x": [1, 2, 3, 4]})
    target = np.array([1, 1, 0, 1])

    scores = evaluate(
        ConstantModel(label=1),
        features,
        target,
        metrics={
            "accuracy": lambda y_true, y_pred: np.mean(y_true == y_pred),
            "errors": lambda y_true, y_pred: np.sum(y_true != y_pred),
        },
    )

    assert scores == {"accuracy": 0.75, "errors": 1.0}
    assert all(type(score) is float for score in scores.values())


def test_evaluate_passes_target_first_and_predictions_second() -> None:
    # Metrics such as precision and recall are not symmetric, so argument order matters.
    features = pd.DataFrame({"x": [1, 2, 3]})
    target = np.array([0, 1, 0])
    received: list[tuple[np.ndarray, np.ndarray]] = []

    def spy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        received.append((y_true, y_pred))
        return 0.0

    evaluate(ConstantModel(label=1), features, target, metrics={"spy": spy})

    [(y_true, y_pred)] = received
    np.testing.assert_array_equal(y_true, target)
    np.testing.assert_array_equal(y_pred, [1, 1, 1])
