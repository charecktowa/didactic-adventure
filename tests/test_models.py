"""Every model in `models/` must pass `validate_model`, with no tests of its own needed.

A model is a package with a `model.py` that exposes `build_model()`, callable without
arguments, returning the model to train.
"""

import numpy as np
import pandas as pd
import pytest

import models
from framework.validation import validate_model

MODEL_NAMES = models.available_models()


@pytest.fixture
def features() -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    size = 100
    return pd.DataFrame(
        {
            "age": rng.normal(40, 10, size),
            "income": rng.normal(1000, 200, size),
            "city": rng.choice(["north", "south", "east"], size),
        }
    )


@pytest.fixture
def target(features: pd.DataFrame) -> np.ndarray:
    return (features["income"] > 1000).astype(int).to_numpy()


def test_finds_the_repository_models() -> None:
    assert {"logistic_regression", "xgboost_classifier"} <= set(MODEL_NAMES)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_is_valid(name: str, features: pd.DataFrame, target: np.ndarray) -> None:
    validate_model(models.build(name), features, target, trainable=True)


def test_rejects_an_unknown_model_name() -> None:
    with pytest.raises(ValueError, match="Unknown model 'framework'"):
        models.build("framework")
