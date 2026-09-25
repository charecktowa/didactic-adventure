from pathlib import Path
from typing import Protocol, Self, runtime_checkable

# TODO: investigate an actual SOLID and agnostic way
import numpy as np
import pandas as pd


@runtime_checkable
class Model(Protocol):
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


class Trainable(Protocol):
    """Optional: implement when the framework has to support training.

    Hyperparameters belong in `__init__`, so `fit` has the same signature for every model.
    """

    def fit(self, features: pd.DataFrame, targets: np.ndarray) -> Self:
        """Train the model using the given features and targets."""
        ...
