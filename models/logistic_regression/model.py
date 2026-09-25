from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from framework.connectors.sklearn import SklearnConnector


def build_model(logistic_params: Mapping[str, Any] | None = None) -> SklearnConnector:
    """Build a logistic regression behind the preprocessing it needs.

    Unlike tree models, it cannot take missing values or unscaled features: numeric columns
    are imputed and scaled, categorical ones imputed and one-hot encoded. Columns are
    picked by dtype: numeric dtypes as numeric, every other column as categorical.
    """
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(
        # SimpleImputer rejects boolean columns: impute them as generic objects.
        FunctionTransformer(np.asarray, kw_args={"dtype": object}),
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    preprocess = make_column_transformer(
        (numeric, make_column_selector(dtype_include="number")),
        (categorical, make_column_selector(dtype_exclude="number")),
    )
    classifier = LogisticRegression(**(logistic_params or {}))
    return SklearnConnector(make_pipeline(preprocess, classifier))
