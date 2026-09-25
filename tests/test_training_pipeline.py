"""The ZenML training pipeline, run end to end on a local, throwaway ZenML store.

Skipped when the `mlops` dependency group is not installed.
"""

from pathlib import Path

import pytest

pytest.importorskip("zenml")

from pipelines import training  # noqa: E402


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
