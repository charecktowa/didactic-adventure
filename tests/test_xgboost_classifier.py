from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from framework.connectors.sklearn import SklearnConnector
from framework.validation import validate_model
from models.xgboost_classifier.model import build_model

NUMERIC = ["age", "income"]
CATEGORICAL = ["city"]
FAST = {"n_estimators": 20, "random_state": 0}


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
    # A signal the model can learn perfectly: only income matters.
    return (features["income"] > 1000).astype(int).to_numpy()


@pytest.fixture
def model(features: pd.DataFrame, target: np.ndarray) -> SklearnConnector:
    return build_model(NUMERIC, CATEGORICAL, xgb_params=FAST).fit(features, target)


def test_is_a_valid_trainable_model(features: pd.DataFrame, target: np.ndarray) -> None:
    validate_model(build_model(NUMERIC, CATEGORICAL, FAST), features, target, trainable=True)


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


def test_passes_xgb_params_to_the_classifier() -> None:
    model = build_model(NUMERIC, xgb_params={"n_estimators": 7, "max_depth": 2})

    classifier = model.estimator[-1]  # type: ignore[index]
    assert (classifier.n_estimators, classifier.max_depth) == (7, 2)


def test_save_then_load_reproduces_predictions(
    model: SklearnConnector, features: pd.DataFrame, tmp_path: Path
) -> None:
    model.save(tmp_path)

    restored = SklearnConnector.load(tmp_path)

    np.testing.assert_array_equal(restored.predict(features), model.predict(features))


def test_keeps_zero_and_missing_numeric_values_apart_with_many_categories() -> None:
    # A high-cardinality column used to make the one-hot output sparse, and XGBoost reads the
    # implicit zeros of a sparse matrix as missing values.
    rng = np.random.default_rng(seed=0)
    size = 1000
    value = rng.choice([np.nan, 0.0, 5.0], size)
    features = pd.DataFrame(
        {"value": value, "city": rng.choice([f"city-{i}" for i in range(100)], size)}
    )
    target = (value == 0).astype(int)

    model = build_model(["value"], ["city"], xgb_params=FAST).fit(features, target)

    np.testing.assert_array_equal(model.predict(features), target)


def test_limits_the_number_of_one_hot_columns() -> None:
    features = pd.DataFrame({"id": [f"id-{i}" for i in range(500)]})
    target = np.arange(500) % 2

    model = build_model([], ["id"], xgb_params={"n_estimators": 1}, max_categories=10)
    model.fit(features, target)

    preprocess = model.estimator[:-1]  # type: ignore[index]
    assert preprocess.transform(features).shape == (500, 10)


def test_fits_when_a_categorical_column_is_entirely_missing(
    features: pd.DataFrame, target: np.ndarray
) -> None:
    features = features.assign(city=np.nan)
    model = build_model(NUMERIC, CATEGORICAL, xgb_params=FAST)

    predictions = model.fit(features, target).predict(features)

    assert predictions.shape == target.shape
