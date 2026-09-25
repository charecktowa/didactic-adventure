"""Deliberately invalid model: a scaler can fit but cannot predict. Do not merge."""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from framework.connectors.sklearn import SklearnConnector


def build_model() -> SklearnConnector:
    """Build a pipeline without an estimator at the end."""
    return SklearnConnector(make_pipeline(StandardScaler()))
