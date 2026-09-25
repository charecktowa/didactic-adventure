from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBClassifier

from framework.connectors.sklearn import SklearnConnector


def build_model(
    xgb_params: Mapping[str, Any] | None = None,
    max_categories: int = 20,
) -> SklearnConnector:
    """Build an XGBoost classifier behind the preprocessing it needs.

    XGBoost handles missing values natively, so numeric columns go through untouched;
    categorical ones are imputed and one-hot encoded. `max_categories` caps the one-hot
    columns per categorical feature; rarer categories are grouped together. Columns are
    picked by dtype: numeric dtypes as numeric, every other column as categorical.
    """
    categorical = make_pipeline(
        # SimpleImputer rejects boolean columns: impute them as generic objects.
        FunctionTransformer(np.asarray, kw_args={"dtype": object}),
        # "missing" becomes a category of its own and keeps all-missing columns.
        SimpleImputer(strategy="constant", fill_value="missing", keep_empty_features=True),
        OneHotEncoder(
            handle_unknown="ignore",
            max_categories=max_categories,
            # Dense: XGBoost reads the implicit zeros of a sparse matrix as missing.
            sparse_output=False,
        ),
    )
    preprocess = make_column_transformer(
        ("passthrough", make_column_selector(dtype_include="number")),
        (categorical, make_column_selector(dtype_exclude="number")),
    )
    classifier = XGBClassifier(**(xgb_params or {}))
    return SklearnConnector(make_pipeline(preprocess, classifier))
