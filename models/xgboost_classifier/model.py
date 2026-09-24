import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

CONFIG_FILE = "config.json"
PIPELINE_FILE = "pipeline.joblib"


class XGBoostClassifier:
    """XGBoost classifier behind a scikit-learn Pipeline that preprocesses the features."""

    def __init__(
        self,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str] = (),
        xgb_params: dict[str, Any] | None = None,
    ) -> None:
        self.config: dict[str, Any] = {
            "numeric_features": list(numeric_features),
            "categorical_features": list(categorical_features),
            "xgb_params": xgb_params or {},
        }
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                # Dense output: XGBoost reads missing entries of a sparse matrix as NaN, not 0.
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        preprocess = ColumnTransformer(
            [
                # XGBoost handles NaN natively, so numeric columns go through untouched.
                ("numeric", "passthrough", self.config["numeric_features"]),
                ("categorical", categorical, self.config["categorical_features"]),
            ]
        )
        return Pipeline(
            [
                ("preprocess", preprocess),
                ("classifier", XGBClassifier(**self.config["xgb_params"])),
            ]
        )

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> Self:
        self.pipeline.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.predict(features))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / CONFIG_FILE).write_text(json.dumps(self.config, indent=2))
        joblib.dump(self.pipeline, directory / PIPELINE_FILE)

    @classmethod
    def load(cls, directory: Path) -> Self:
        model = cls(**json.loads((directory / CONFIG_FILE).read_text()))
        # joblib uses pickle under the hood: only load directories you trust.
        model.pipeline = joblib.load(directory / PIPELINE_FILE)
        return model
