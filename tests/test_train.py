import sys
from pathlib import Path

import pytest

from models.xgboost_classifier import train
from models.xgboost_classifier.model import XGBoostClassifier


def test_load_data_returns_aligned_features_and_binary_target() -> None:
    features, target = train.load_data()

    assert len(features) == len(target) > 0
    assert set(target) == {0, 1}


def test_main_prints_metrics_and_saves_a_loadable_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "model"
    monkeypatch.setattr(sys, "argv", ["train", "--output", str(output)])

    train.main()

    stdout = capsys.readouterr().out
    for metric in ("accuracy", "precision", "recall", "f1"):
        assert metric in stdout
    features, _ = train.load_data()
    restored = XGBoostClassifier.load(output)
    assert restored.predict(features.iloc[:5]).shape == (5,)


def test_main_without_output_does_not_write_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train"])

    train.main()

    assert list(tmp_path.iterdir()) == []
