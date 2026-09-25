from collections.abc import Mapping, Sequence
from typing import Any

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from framework.connectors.sklearn import SklearnConnector


def build_model(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = (),
    logistic_params: Mapping[str, Any] | None = None,
) -> SklearnConnector:
    """Build a logistic regression behind the preprocessing it needs.

    Unlike tree models, it cannot take missing values or unscaled features: numeric columns
    are imputed and scaled, categorical ones imputed and one-hot encoded.
    """
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
    )
    preprocess = make_column_transformer(
        (numeric, list(numeric_features)),
        (categorical, list(categorical_features)),
    )
    classifier = LogisticRegression(**(logistic_params or {}))
    return SklearnConnector(make_pipeline(preprocess, classifier))
