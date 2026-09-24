from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from framework.contracts.model import Model, Trainable
from framework.evaluation import evaluate
from models.xgboost_classifier.model import XGBoostClassifier


@pytest.fixture
def data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    size = 200
    features = pd.DataFrame(
        {
            "age": rng.normal(40, 10, size),
            "income": rng.normal(1000, 200, size),
            "city": rng.choice(["a", "b", "c"], size),
        }
    )
    features.loc[::10, "age"] = np.nan
    features.loc[::15, "city"] = np.nan
    target = (features["income"] > 1000).astype(int).to_numpy()
    return features, target


def make_model() -> XGBoostClassifier:
    return XGBoostClassifier(
        numeric_features=["age", "income"],
        categorical_features=["city"],
        xgb_params={"n_estimators": 10, "random_state": 0},
    )


def test_fulfils_contract() -> None:
    model = make_model()
    assert isinstance(model, Model)
    assert isinstance(model, Trainable)


def test_fit_predict(data: tuple[pd.DataFrame, np.ndarray]) -> None:
    features, target = data
    predictions = make_model().fit(features, target).predict(features)
    assert predictions.shape == target.shape
    assert set(np.unique(predictions)) <= {0, 1}


def test_save_load_roundtrip(data: tuple[pd.DataFrame, np.ndarray], tmp_path: Path) -> None:
    features, target = data
    model = make_model().fit(features, target)

    model.save(tmp_path / "model")
    restored = XGBoostClassifier.load(tmp_path / "model")

    assert restored.config == model.config
    np.testing.assert_array_equal(restored.predict(features), model.predict(features))


def test_evaluate(data: tuple[pd.DataFrame, np.ndarray]) -> None:
    features, target = data
    model = make_model().fit(features, target)

    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float((y_true == y_pred).mean())

    assert evaluate(model, features, target, {"accuracy": accuracy})["accuracy"] > 0.9
