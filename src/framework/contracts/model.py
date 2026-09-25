from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Model(Protocol):
    """What the framework needs from a trained model, whatever library is behind it."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return predictions for the supplied feature rows."""
        ...

    def save(self, directory: Path) -> None:
        """Persist the model inside `directory`; the model decides which files to write."""
        ...

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Restore a model from files in ``directory``."""
        ...


@runtime_checkable
class Trainable(Protocol):
    """Optional: implement it when the framework has to launch training.

    Hyperparameters belong in `__init__`, so `fit` has the same signature for every model.
    """

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> Self:
        """Train on labeled features and return the fitted model."""
        ...
