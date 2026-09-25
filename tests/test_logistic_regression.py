from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from framework.connectors.sklearn import SklearnConnector
from framework.validation import validate_model
from models.logistic_regression.model import build_model

NUMERIC = ["age", "income"]
CATEGORICAL = ["city"]


@pytest.fixture
def features() -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    size = 200
    return pd.DataFrame(
        {
            "age": rng.normal(40, 10, size),
            "income": rng.normal(1000, 200, size),
            "city": rng.choice(["north", "south", "east"], size),
        }
    )


@pytest.fixture
def target(features: pd.DataFrame) -> np.ndarray:
    # A linear signal the model can learn: only income matters.
    return (features["income"] > 1000).astype(int).to_numpy()


@pytest.fixture
def model(features: pd.DataFrame, target: np.ndarray) -> SklearnConnector:
    return build_model(NUMERIC, CATEGORICAL).fit(features, target)


def test_is_a_valid_trainable_model(features: pd.DataFrame, target: np.ndarray) -> None:
    validate_model(build_model(NUMERIC, CATEGORICAL), features, target, trainable=True)


def test_learns_a_simple_signal(
    model: SklearnConnector, features: pd.DataFrame, target: np.ndarray
) -> None:
    accuracy = np.mean(model.predict(features) == target)

    assert accuracy > 0.95


@pytest.mark.parametrize(
    "row",
    [
        pytest.param({"age": 30.0, "income": 1500.0, "city": "west"}, id="unseen-category"),
        pytest.param({"age": np.nan, "income": 1500.0, "city": "north"}, id="missing-numeric"),
        pytest.param({"age": 30.0, "income": 1500.0, "city": np.nan}, id="missing-categorical"),
    ],
)
def test_predicts_rows_not_seen_during_training(
    model: SklearnConnector, row: dict[str, object]
) -> None:
    prediction = model.predict(pd.DataFrame([row]))

    assert prediction.tolist() == [1]


def test_passes_logistic_params_to_the_classifier() -> None:
    model = build_model(NUMERIC, logistic_params={"C": 0.1, "max_iter": 500})

    classifier = model.estimator[-1]  # type: ignore[index]
    assert (classifier.C, classifier.max_iter) == (0.1, 500)


def test_save_then_load_reproduces_predictions(
    model: SklearnConnector, features: pd.DataFrame, tmp_path: Path
) -> None:
    model.save(tmp_path)

    restored = SklearnConnector.load(tmp_path)

    np.testing.assert_array_equal(restored.predict(features), model.predict(features))
