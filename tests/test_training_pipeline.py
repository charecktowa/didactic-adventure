"""The ZenML training pipeline and its model materializer, on a local, throwaway ZenML store.

Not collected when the `mlops` dependency group is not installed (see `conftest.py`).
"""

import json
import os
import re
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from framework.connectors.sklearn import SklearnConnector
from framework.contracts.model import Model
from materializers.model import MODEL_CLASS_FILE, MODEL_DIR, ModelMaterializer
from pipelines import training


@pytest.fixture(autouse=True)
def isolated_zenml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the test's runs and artifacts out of the developer's own ZenML store.
    monkeypatch.setenv("ZENML_CONFIG_PATH", str(tmp_path / "zenml"))
    monkeypatch.setenv("ZENML_ANALYTICS_OPT_IN", "false")
    monkeypatch.chdir(tmp_path)


def test_trains_and_evaluates_a_repository_model() -> None:
    metrics = training.run_training("logistic_regression")

    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert metrics["accuracy"] > 0.9


class ThresholdRule:
    """A model that fulfils the `Model` protocol without inheriting from any connector."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (features["x"] > self.threshold).astype(int).to_numpy()

    def save(self, directory: Path) -> None:
        (directory / "rule.json").write_text(json.dumps({"threshold": self.threshold}))

    @classmethod
    def load(cls, directory: Path) -> Self:
        return cls(**json.loads((directory / "rule.json").read_text()))


def test_stores_the_model_in_its_own_format_instead_of_a_pickle() -> None:
    run = training.training_pipeline(model_name="logistic_regression")
    assert run is not None
    artifact = run.steps["train"].outputs["model"][0]

    # What save() wrote, loadable without ZenML.
    restored = SklearnConnector.load(Path(artifact.uri) / MODEL_DIR)

    features, _ = load_breast_cancer(return_X_y=True, as_frame=True)
    np.testing.assert_array_equal(artifact.load().predict(features), restored.predict(features))


def test_round_trips_a_model_that_is_not_a_connector(tmp_path: Path) -> None:
    features = pd.DataFrame({"x": [0.1, 0.9]})
    uri = str(tmp_path / "artifact")

    ModelMaterializer(uri).save(ThresholdRule(threshold=0.5))
    restored = ModelMaterializer(uri).load(Model)

    assert isinstance(restored, ThresholdRule)
    assert restored.predict(features).tolist() == [0, 1]


def test_refuses_to_load_as_an_unrelated_model_class(tmp_path: Path) -> None:
    uri = str(tmp_path / "artifact")
    ModelMaterializer(uri).save(ThresholdRule(threshold=0.5))

    with pytest.raises(TypeError, match="ThresholdRule"):
        ModelMaterializer(uri).load(SklearnConnector)


@pytest.mark.parametrize(
    ("content", "error", "message"),
    [
        pytest.param("{broken", ValueError, "does not name the class", id="malformed-json"),
        pytest.param("{}", ValueError, "does not name the class", id="missing-fields"),
        pytest.param("[]", ValueError, "does not name the class", id="not-an-object"),
        pytest.param(
            json.dumps({"module": 1, "qualname": ["x"]}),
            ValueError,
            "does not name the class",
            id="fields-not-strings",
        ),
        pytest.param(
            json.dumps({"module": "models.gone", "qualname": "Model"}),
            ValueError,
            "models.gone.Model, which cannot be found",
            id="module-moved",
        ),
        pytest.param(
            json.dumps({"module": "framework.connectors.sklearn", "qualname": "Renamed"}),
            ValueError,
            "framework.connectors.sklearn.Renamed, which cannot be found",
            id="class-renamed",
        ),
        pytest.param(
            json.dumps({"module": "os.path", "qualname": "join"}),
            TypeError,
            "os.path.join, which is not a Model class",
            id="not-a-class",
        ),
        pytest.param(
            json.dumps({"module": "builtins", "qualname": "dict"}),
            TypeError,
            "builtins.dict, which is not a Model class",
            id="not-a-model",
        ),
    ],
)
def test_rejects_an_artifact_that_does_not_name_a_model_class(
    tmp_path: Path, content: str, error: type[Exception], message: str
) -> None:
    uri = str(tmp_path / "artifact")
    ModelMaterializer(uri).save(ThresholdRule(threshold=0.5))
    with open(os.path.join(uri, MODEL_CLASS_FILE), "w") as file:
        file.write(content)

    with pytest.raises(error, match=re.escape(message)):
        ModelMaterializer(uri).load(Model)


def test_rejects_an_artifact_without_its_class_file(tmp_path: Path) -> None:
    uri = str(tmp_path / "artifact")
    ModelMaterializer(uri).save(ThresholdRule(threshold=0.5))
    os.remove(os.path.join(uri, MODEL_CLASS_FILE))

    with pytest.raises(ValueError, match="does not name the class"):
        ModelMaterializer(uri).load(Model)
