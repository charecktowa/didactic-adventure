"""Contracts the framework depends on, defined by capability rather than by ML library.

Models fulfil them structurally: inheriting from them is not required.
"""

from pathlib import Path
from typing import Protocol, Self, runtime_checkable

# TODO: investigate an actual SOLID and agnostic way
import numpy as np
import pandas as pd


@runtime_checkable
class Model(Protocol):
    """A model the framework can predict with and persist. Every model must fulfil it."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict the output based on the given features."""
        ...

    def save(self, directory: Path) -> None:
        """Save the model to the specified directory."""
        ...

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Load the model from the specified directory."""
        ...


@runtime_checkable
class Trainable(Protocol):
    """Optional capability: implement it when the framework has to train the model.

    Hyperparameters belong in `__init__`, so `fit` has the same signature for every model.
    """

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> Self:
        """Train the model using the given features and target."""
        ...
