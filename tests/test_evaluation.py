from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from framework.evaluation import evaluate


def test_evaluate_predicts_once_and_passes_targets_before_predictions():
    features = pd.DataFrame({"value": [1, 2, 3]})
    target = np.array([0, 1, 0])
    predictions = np.array([0, 0, 1])
    model = Mock()
    model.predict.return_value = predictions
    count_correct = Mock(return_value=np.int64(1))
    proportion_correct = Mock(return_value=np.float32(1 / 3))

    scores = evaluate(
        model,
        features,
        target,
        {"count_correct": count_correct, "proportion_correct": proportion_correct},
    )

    model.predict.assert_called_once_with(features)
    for metric in (count_correct, proportion_correct):
        metric.assert_called_once()
        actual_target, actual_predictions = metric.call_args.args
        assert actual_target is target
        assert actual_predictions is predictions
    assert list(scores) == ["count_correct", "proportion_correct"]
    assert scores == {"count_correct": 1.0, "proportion_correct": float(np.float32(1 / 3))}
    assert all(isinstance(score, float) for score in scores.values())


def test_evaluate_with_no_metrics_returns_empty_scores():
    model = Mock()
    model.predict.return_value = np.array([1])
    features = pd.DataFrame({"value": [5]})

    assert evaluate(model, features, np.array([1]), {}) == {}
    model.predict.assert_called_once_with(features)


def test_evaluate_propagates_metric_errors_without_retrying_prediction():
    model = Mock()
    model.predict.return_value = np.array([1])

    def failing_metric(_target: np.ndarray, _predictions: np.ndarray) -> float:
        raise ValueError("invalid metric input")

    with pytest.raises(ValueError, match="invalid metric input"):
        evaluate(model, pd.DataFrame({"value": [1]}), np.array([1]), {"broken": failing_metric})

    model.predict.assert_called_once()
