import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from framework.contracts.model import Model, Trainable
from models.xgboost_classifier.model import CONFIG_FILE, PIPELINE_FILE, XGBoostClassifier

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


def test_fulfils_model_and_trainable_contracts(model: XGBoostClassifier) -> None:
    contract: Model = model  # checked statically by mypy
    trainable: Trainable = model

    assert isinstance(contract, Model)
    assert isinstance(trainable, Trainable)


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


def test_loads_a_model_saved_before_metadata_existed(
    model: XGBoostClassifier, features: pd.DataFrame, tmp_path: Path
) -> None:
    # The layout written by save() before ModelConnector: no metadata.json.
    (tmp_path / CONFIG_FILE).write_text(json.dumps(model.config))
    joblib.dump(model.pipeline, tmp_path / PIPELINE_FILE)

    with pytest.warns(UserWarning, match="legacy"):
        restored = XGBoostClassifier.load(tmp_path)

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

    model = XGBoostClassifier(
        numeric_features=["value"],
        categorical_features=["city"],
        xgb_params={"n_estimators": 20, "random_state": 0},
    ).fit(features, target)

    np.testing.assert_array_equal(model.predict(features), target)


def test_limits_the_number_of_one_hot_columns() -> None:
    features = pd.DataFrame({"id": [f"id-{i}" for i in range(500)]})
    target = np.arange(500) % 2

    model = XGBoostClassifier(
        numeric_features=[],
        categorical_features=["id"],
        xgb_params={"n_estimators": 1, "random_state": 0},
        max_categories=10,
    ).fit(features, target)

    assert model.pipeline[:-1].transform(features).shape == (500, 10)


def test_fits_when_a_categorical_column_is_entirely_missing(
    features: pd.DataFrame, target: np.ndarray
) -> None:
    features = features.assign(city=np.nan)
    model = XGBoostClassifier(numeric_features=NUMERIC, categorical_features=CATEGORICAL)

    predictions = model.fit(features, target).predict(features)

    assert predictions.shape == target.shape


def test_rejects_xgb_params_that_cannot_be_saved() -> None:
    with pytest.raises(ValueError, match="JSON"):
        XGBoostClassifier(numeric_features=NUMERIC, xgb_params={"eval_metric": len})


@pytest.mark.parametrize("max_categories", [0, -1])
def test_rejects_max_categories_below_one(max_categories: int) -> None:
    with pytest.raises(ValueError, match="max_categories"):
        XGBoostClassifier(numeric_features=NUMERIC, max_categories=max_categories)
