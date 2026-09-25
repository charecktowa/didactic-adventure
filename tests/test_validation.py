"""`validate_model` against XGBoost wrappers written by hand, without the connector base.

Each broken wrapper shows a mistake that a protocol check alone would miss or report late.
"""

import re
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from framework.validation import InvalidModelError, validate_model
from models.xgboost_classifier.model import XGBoostClassifier


@pytest.fixture
def features() -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    return pd.DataFrame({"x": rng.normal(size=60)})


@pytest.fixture
def target(features: pd.DataFrame) -> np.ndarray:
    return (features["x"] > 0).astype(int).to_numpy()


class HandMadeXGBoost:
    """A valid wrapper that does not inherit from anything: the protocols are enough."""

    def __init__(self) -> None:
        self.estimator = XGBClassifier(n_estimators=5, random_state=0)

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> Self:
        self.estimator.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(features))

    def save(self, directory: Path) -> None:
        self.estimator.save_model(directory / "model.json")

    @classmethod
    def load(cls, directory: Path) -> Self:
        model = cls()
        model.estimator.load_model(directory / "model.json")
        return model


class WithoutPredict:
    def fit(self, features: pd.DataFrame, target: np.ndarray) -> Self:
        return self

    def save(self, directory: Path) -> None: ...

    @classmethod
    def load(cls, directory: Path) -> Self:
        return cls()


class WithoutFit(HandMadeXGBoost):
    """Nothing ever trains the estimator it predicts with."""

    fit = None  # type: ignore[assignment]


class PredictWithoutFeatures(HandMadeXGBoost):
    def predict(self) -> np.ndarray:  # type: ignore[override]
        return np.zeros(1)


class OnePredictionForAllRows(HandMadeXGBoost):
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return super().predict(features)[:1]


class UninspectablePredict(HandMadeXGBoost):
    # A builtin callable without an inspectable signature.
    predict = staticmethod(ValueError)  # type: ignore[assignment]


class ForgetsTrainingOnLoad(HandMadeXGBoost):
    @classmethod
    def load(cls, directory: Path) -> Self:
        return cls()


def test_accepts_the_repository_connector(features: pd.DataFrame, target: np.ndarray) -> None:
    model = XGBoostClassifier(numeric_features=["x"], xgb_params={"n_estimators": 5})

    assert validate_model(model, features, target, trainable=True) is model


def test_accepts_a_model_that_does_not_inherit_from_anything(
    features: pd.DataFrame, target: np.ndarray
) -> None:
    model = HandMadeXGBoost()

    assert validate_model(model, features, target, trainable=True) is model


def test_does_not_refit_a_model_that_is_not_validated_for_training(
    features: pd.DataFrame, target: np.ndarray
) -> None:
    pretrained = HandMadeXGBoost().fit(features, target)
    booster_before = pretrained.estimator.get_booster().save_raw()

    validate_model(pretrained, features, 1 - target, trainable=False)

    assert pretrained.estimator.get_booster().save_raw() == booster_before


def test_accepts_contract_methods_with_other_parameter_names(
    features: pd.DataFrame, target: np.ndarray
) -> None:
    class SklearnStyleNames(HandMadeXGBoost):
        def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:  # noqa: N803
            return super().fit(X, y)

    model = SklearnStyleNames()

    assert validate_model(model, features, target, trainable=True) is model


@pytest.mark.parametrize(
    ("model", "trainable", "problem"),
    [
        pytest.param(WithoutPredict(), False, "missing method predict()", id="no-predict"),
        pytest.param(WithoutFit(), True, "missing method fit() required by Trainable", id="no-fit"),
        pytest.param(WithoutFit(), False, "raised NotFittedError", id="no-fit-used-as-pretrained"),
        pytest.param(
            PredictWithoutFeatures(), True, "must accept (features)", id="wrong-signature"
        ),
        pytest.param(
            UninspectablePredict(), False, "no inspectable signature", id="uninspectable-signature"
        ),
        pytest.param(OnePredictionForAllRows(), True, "one value per row", id="wrong-output-shape"),
        pytest.param(ForgetsTrainingOnLoad(), True, "raised NotFittedError", id="breaks-on-load"),
    ],
)
def test_rejects_broken_models(
    features: pd.DataFrame, target: np.ndarray, model: object, trainable: bool, problem: str
) -> None:
    with pytest.raises(InvalidModelError, match=re.escape(problem)):
        validate_model(model, features, target, trainable=trainable)


def test_reports_every_problem_at_once(features: pd.DataFrame, target: np.ndarray) -> None:
    class Empty:
        pass

    with pytest.raises(InvalidModelError) as error:
        validate_model(Empty(), features, target, trainable=True)

    assert error.value.problems == [
        "missing method load() required by Model",
        "missing method predict() required by Model",
        "missing method save() required by Model",
        "missing method fit() required by Trainable",
    ]
