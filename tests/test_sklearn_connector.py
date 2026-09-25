import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from framework.connectors.base import METADATA_FILE
from framework.connectors.sklearn import ESTIMATOR_FILE, SklearnConnector
from framework.contracts.model import Model, Trainable
from framework.validation import InvalidModelError, validate_model


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame({"x": np.linspace(0, 1, 20)})


@pytest.fixture
def target(features: pd.DataFrame) -> np.ndarray:
    return (features["x"] > 0.5).astype(int).to_numpy()


def test_fulfils_model_and_trainable_contracts() -> None:
    connector = SklearnConnector(LogisticRegression())
    contract: Model = connector  # checked statically by mypy
    trainable: Trainable = connector

    assert isinstance(contract, Model)
    assert isinstance(trainable, Trainable)


def test_fit_trains_the_estimator_and_returns_the_same_instance(
    features: pd.DataFrame, target: np.ndarray
) -> None:
    connector = SklearnConnector(DecisionTreeClassifier(random_state=0))

    assert connector.fit(features, target) is connector
    np.testing.assert_array_equal(connector.predict(features), target)


def test_save_writes_metadata_and_estimator(
    features: pd.DataFrame, target: np.ndarray, tmp_path: Path
) -> None:
    SklearnConnector(LogisticRegression()).fit(features, target).save(tmp_path)

    metadata = json.loads((tmp_path / METADATA_FILE).read_text())
    assert metadata["connector"] == "framework.connectors.sklearn.SklearnConnector"
    assert metadata["library"] == "sklearn"
    assert (tmp_path / ESTIMATOR_FILE).exists()


def test_load_restores_the_fitted_estimator(
    features: pd.DataFrame, target: np.ndarray, tmp_path: Path
) -> None:
    connector = SklearnConnector(LogisticRegression()).fit(features, target)
    connector.save(tmp_path)

    restored = SklearnConnector.load(tmp_path)

    np.testing.assert_array_equal(restored.predict(features), connector.predict(features))


def test_a_classifier_passes_validation(features: pd.DataFrame, target: np.ndarray) -> None:
    connector = SklearnConnector(LogisticRegression())

    assert validate_model(connector, features, target, trainable=True) is connector


def test_an_unfitted_estimator_fails_validation_as_a_pretrained_model(
    features: pd.DataFrame, target: np.ndarray
) -> None:
    with pytest.raises(InvalidModelError, match="NotFittedError"):
        validate_model(SklearnConnector(LogisticRegression()), features, target)


def test_an_estimator_without_predict_fails_validation(
    features: pd.DataFrame, target: np.ndarray
) -> None:
    # A transformer fits but cannot predict: only running it reveals the problem.
    connector = SklearnConnector(StandardScaler())

    with pytest.raises(InvalidModelError, match="AttributeError"):
        validate_model(connector, features, target, trainable=True)
