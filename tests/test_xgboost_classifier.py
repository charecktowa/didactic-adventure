import json

import numpy as np
import pandas as pd
import pytest

from framework.contracts.model import Model
from models.xgboost_classifier.model import CONFIG_FILE, PIPELINE_FILE, XGBoostClassifier


@pytest.fixture
def training_data():
    values = np.arange(-10, 10, dtype=float)
    features = pd.DataFrame(
        {
            "measurement": values,
            "category": ["even" if value % 2 == 0 else "odd" for value in values],
        }
    )
    features.loc[0, "measurement"] = np.nan
    features.loc[1, "category"] = np.nan
    target = (values >= 0).astype(int)
    return features, target


@pytest.fixture
def fitted_model(training_data):
    features, target = training_data
    model = XGBoostClassifier(
        numeric_features=["measurement"],
        categorical_features=["category"],
        xgb_params={
            "n_estimators": 25,
            "max_depth": 2,
            "learning_rate": 0.5,
            "n_jobs": 1,
            "random_state": 0,
        },
    )
    assert model.fit(features, target) is model
    return model


def test_preprocessing_preserves_numeric_nan_and_handles_missing_and_unseen_categories():
    model = XGBoostClassifier(["measurement"], ["category"], {"n_estimators": 2})
    features = pd.DataFrame(
        {"measurement": [1.0, 2.0, 3.0, 4.0], "category": ["red", "red", "blue", np.nan]}
    )
    preprocess = model.pipeline.named_steps["preprocess"]
    preprocess.fit(features)

    missing = preprocess.transform(pd.DataFrame({"measurement": [np.nan], "category": [np.nan]}))
    unseen = preprocess.transform(pd.DataFrame({"measurement": [5.0], "category": ["green"]}))

    assert isinstance(missing, np.ndarray)
    assert np.isnan(missing[0, 0])
    np.testing.assert_array_equal(missing[0, 1:], [0.0, 1.0])  # "red" is the mode.
    np.testing.assert_array_equal(unseen[0, 1:], [0.0, 0.0])


def test_fit_predicts_with_mixed_features_and_satisfies_model_contract(fitted_model, training_data):
    features, target = training_data

    predictions = fitted_model.predict(features)
    assert isinstance(fitted_model, Model)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == target.shape
    assert set(np.unique(predictions)) <= {0, 1}
    assert np.mean(predictions == target) >= 0.9

    unseen = pd.DataFrame({"measurement": [np.nan, 3.0], "category": ["new", "even"]})
    assert fitted_model.predict(unseen).shape == (2,)


def test_save_load_round_trip_preserves_config_and_predictions(fitted_model, training_data, tmp_path):
    features, _target = training_data
    directory = tmp_path / "nested" / "model"

    fitted_model.save(directory)
    restored = XGBoostClassifier.load(directory)

    assert (directory / CONFIG_FILE).is_file()
    assert (directory / PIPELINE_FILE).is_file()
    assert json.loads((directory / CONFIG_FILE).read_text()) == fitted_model.config
    assert restored.config == fitted_model.config
    np.testing.assert_array_equal(restored.predict(features), fitted_model.predict(features))


def test_load_requires_saved_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        XGBoostClassifier.load(tmp_path)
