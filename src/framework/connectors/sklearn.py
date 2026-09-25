"""Connector for any estimator that follows the scikit-learn API."""

from pathlib import Path
from typing import Any, Protocol, Self

import joblib
import numpy as np
import pandas as pd

from framework.connectors.base import ModelConnector

ESTIMATOR_FILE = "estimator.joblib"


class Estimator(Protocol):
    """The scikit-learn API the connector relies on; estimators need not inherit from sklearn."""

    def fit(self, features: Any, target: Any, /) -> Any: ...

    def predict(self, features: Any, /) -> Any: ...


class SklearnConnector(ModelConnector):
    """Wraps a scikit-learn estimator or Pipeline, which carries its own hyperparameters."""

    library = "sklearn"

    def __init__(self, estimator: Estimator) -> None:
        self.estimator = estimator

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> Self:
        """Fit the wrapped estimator and return this connector."""
        self.estimator.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict with the wrapped estimator."""
        return np.asarray(self.estimator.predict(features))

    def _save_artifacts(self, directory: Path) -> None:
        joblib.dump(self.estimator, directory / ESTIMATOR_FILE)

    @classmethod
    def _load_artifacts(cls, directory: Path) -> Self:
        # joblib uses pickle under the hood: only load directories you trust.
        return cls(joblib.load(directory / ESTIMATOR_FILE))
