from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from framework.contracts.model import Model
from models.xgboost_classifier.model import XGBoostClassifier

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
    # A signal the model can learn perfectly: only income matters.
    return (features["income"] > 1000).astype(int).to_numpy()


@pytest.fixture
def model(features: pd.DataFrame, target: np.ndarray) -> XGBoostClassifier:
    return XGBoostClassifier(
        numeric_features=NUMERIC,
        categorical_features=CATEGORICAL,
        xgb_params={"n_estimators": 20, "random_state": 0},
    ).fit(features, target)


def test_fulfils_model_contract(model: XGBoostClassifier) -> None:
    contract: Model = model  # checked statically by mypy

    assert isinstance(contract, Model)


def test_fit_returns_the_same_instance(features: pd.DataFrame, target: np.ndarray) -> None:
    model = XGBoostClassifier(numeric_features=NUMERIC, categorical_features=CATEGORICAL)

    assert model.fit(features, target) is model


def test_predict_returns_one_binary_label_per_row(
    model: XGBoostClassifier, features: pd.DataFrame
) -> None:
    predictions = model.predict(features)

    assert predictions.shape == (len(features),)
    assert set(np.unique(predictions)) <= {0, 1}


def test_learns_a_simple_signal(
    model: XGBoostClassifier, features: pd.DataFrame, target: np.ndarray
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
    model: XGBoostClassifier, row: dict[str, object]
) -> None:
    prediction = model.predict(pd.DataFrame([row]))

    assert prediction.tolist() == [1]


def test_save_then_load_reproduces_config_and_predictions(
    model: XGBoostClassifier, features: pd.DataFrame, tmp_path: Path
) -> None:
    directory = tmp_path / "nested" / "model"

    model.save(directory)
    restored = XGBoostClassifier.load(directory)

    assert restored.config == model.config
    np.testing.assert_array_equal(restored.predict(features), model.predict(features))
