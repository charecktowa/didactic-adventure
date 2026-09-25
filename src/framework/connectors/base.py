"""Base class that makes writing a connector for an ML library easier."""

import json
from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pandas as pd

METADATA_FILE = "metadata.json"


class ModelConnector(ABC):
    """Adapts an ML library to the `Model` contract.

    Inheriting from it is optional: the framework only depends on the `Model` protocol. It
    gives connector authors shared persistence with metadata, and fails at instantiation when
    an abstract method is missing. Connectors that train also define `fit`, which makes them
    `Trainable`.
    """

    library: ClassVar[str]
    """Import name of the wrapped library, e.g. ``"xgboost"``; recorded in the metadata."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict the output based on the given features."""

    @abstractmethod
    def _save_artifacts(self, directory: Path) -> None:
        """Write the library-specific files into an existing ``directory``."""

    @classmethod
    @abstractmethod
    def _load_artifacts(cls, directory: Path) -> Self:
        """Rebuild the connector from the files written by `_save_artifacts`."""

    def save(self, directory: Path) -> None:
        """Save the metadata and the connector's artifacts into ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "connector": self._qualified_name(),
            "library": self.library,
            "library_version": getattr(import_module(self.library), "__version__", None),
        }
        (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2))
        self._save_artifacts(directory)

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Load a model saved by this same connector from ``directory``."""
        metadata = json.loads((directory / METADATA_FILE).read_text())
        if metadata["connector"] != cls._qualified_name():
            raise ValueError(
                f"{directory} was saved by {metadata['connector']}, "
                f"it cannot be loaded with {cls._qualified_name()}"
            )
        return cls._load_artifacts(directory)

    @classmethod
    def _qualified_name(cls) -> str:
        return f"{cls.__module__}.{cls.__qualname__}"
