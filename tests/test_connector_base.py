import json
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import pytest

from framework.connectors.base import METADATA_FILE, ModelConnector
from framework.contracts.model import Model, Trainable


class ThresholdRule(ModelConnector):
    """A pretrained connector: predicts with a fixed rule and has no `fit`."""

    library = "numpy"

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (features["x"] > self.threshold).astype(int).to_numpy()

    def _save_artifacts(self, directory: Path) -> None:
        (directory / "rule.json").write_text(json.dumps({"threshold": self.threshold}))

    @classmethod
    def _load_artifacts(cls, directory: Path) -> Self:
        return cls(**json.loads((directory / "rule.json").read_text()))


class OtherRule(ThresholdRule):
    pass


def test_connector_without_fit_is_a_model_but_not_trainable() -> None:
    connector: Model = ThresholdRule(threshold=0.5)  # checked statically by mypy

    assert isinstance(connector, Model)
    assert not isinstance(connector, Trainable)


def test_cannot_instantiate_a_connector_that_misses_an_abstract_method() -> None:
    class ForgotPredict(ModelConnector):
        library = "numpy"

        def _save_artifacts(self, directory: Path) -> None: ...

        @classmethod
        def _load_artifacts(cls, directory: Path) -> Self:
            return cls()

    with pytest.raises(TypeError, match="predict"):
        ForgotPredict()  # type: ignore[abstract]


def test_save_writes_metadata_next_to_the_artifacts(tmp_path: Path) -> None:
    ThresholdRule(threshold=0.5).save(tmp_path / "model")

    metadata = json.loads((tmp_path / "model" / METADATA_FILE).read_text())
    assert metadata == {
        "connector": f"{__name__}.ThresholdRule",
        "library": "numpy",
        "library_version": np.__version__,
    }
    assert (tmp_path / "model" / "rule.json").exists()


def test_load_restores_the_connector(tmp_path: Path) -> None:
    features = pd.DataFrame({"x": [0.1, 0.9]})
    ThresholdRule(threshold=0.5).save(tmp_path)

    restored = ThresholdRule.load(tmp_path)

    assert restored.threshold == 0.5
    assert restored.predict(features).tolist() == [0, 1]


def test_load_rejects_a_model_saved_by_another_connector(tmp_path: Path) -> None:
    OtherRule(threshold=0.5).save(tmp_path)

    with pytest.raises(ValueError, match="OtherRule"):
        ThresholdRule.load(tmp_path)


def test_load_rejects_metadata_without_a_connector(tmp_path: Path) -> None:
    ThresholdRule(threshold=0.5).save(tmp_path)
    (tmp_path / METADATA_FILE).write_text("{}")

    with pytest.raises(ValueError, match="does not say which connector"):
        ThresholdRule.load(tmp_path)
