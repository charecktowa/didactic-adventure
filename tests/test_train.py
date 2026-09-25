import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from models.xgboost_classifier import train


def test_load_data_returns_aligned_bundled_features_and_labels():
    features, target = train.load_data()

    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, np.ndarray)
    assert features.shape == (569, 30)
    assert target.shape == (569,)
    assert set(np.unique(target)) == {0, 1}
    assert features.columns.is_unique


@pytest.mark.parametrize("save_output", [False, True])
def test_main_trains_scores_and_optionally_saves(monkeypatch, capsys, tmp_path, save_output):
    features = pd.DataFrame({"measurement": [1.0, 2.0, 3.0, 4.0]})
    target = np.array([0, 0, 1, 1])
    x_train, x_test = features.iloc[:2], features.iloc[2:]
    y_train, y_test = target[:2], target[2:]
    split = Mock(return_value=(x_train, x_test, y_train, y_test))
    model = Mock()
    model.fit.return_value = model
    classifier = Mock(return_value=model)
    evaluation = Mock(return_value={"accuracy": 0.75, "precision": 0.5})
    output = tmp_path / "trained" if save_output else None
    argv = ["train", "--output", str(output)] if output else ["train"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(train, "load_data", lambda: (features, target))
    monkeypatch.setattr(train, "train_test_split", split)
    monkeypatch.setattr(train, "XGBoostClassifier", classifier)
    monkeypatch.setattr(train, "evaluate", evaluation)

    train.main()

    split.assert_called_once_with(
        features, target, test_size=0.2, stratify=target, random_state=train.RANDOM_STATE
    )
    classifier.assert_called_once_with(
        numeric_features=["measurement"],
        xgb_params={
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": train.RANDOM_STATE,
        },
    )
    model.fit.assert_called_once_with(x_train, y_train)
    evaluated_model, evaluated_features, evaluated_target, metrics = evaluation.call_args.args
    assert evaluated_model is model
    assert evaluated_features is x_test
    assert evaluated_target is y_test
    assert list(metrics) == ["accuracy", "precision", "recall", "f1"]
    assert metrics["accuracy"] is train.accuracy_score
    assert metrics["precision"] is train.precision_score
    assert metrics["recall"] is train.recall_score
    assert metrics["f1"] is train.f1_score
    output_lines = capsys.readouterr().out.splitlines()
    assert output_lines[:2] == ["accuracy   0.7500", "precision  0.5000"]
    if output:
        model.save.assert_called_once_with(Path(output))
        assert output_lines[2] == f"Model saved to {output}"
    else:
        model.save.assert_not_called()
        assert len(output_lines) == 2
