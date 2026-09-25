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
        """Configure the feature columns and XGBoost parameters for the pipeline."""
        self.config: dict[str, Any] = {
            "numeric_features": list(numeric_features),
            "categorical_features": list(categorical_features),
            "xgb_params": xgb_params or {},
        }
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        """Build preprocessing steps and the XGBoost classifier from the config."""
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                # Keep sparse output to avoid materializing large one-hot matrices in memory.
                ("encode", OneHotEncoder(handle_unknown="ignore")),
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
        """Fit the pipeline on labeled features and return this model."""
        self.pipeline.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the supplied features."""
        return np.asarray(self.pipeline.predict(features))

    def save(self, directory: Path) -> None:
        """Write the model config and fitted pipeline into ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / CONFIG_FILE).write_text(json.dumps(self.config, indent=2))
        joblib.dump(self.pipeline, directory / PIPELINE_FILE)

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Restore a model from a trusted directory containing saved artifacts."""
        model = cls(**json.loads((directory / CONFIG_FILE).read_text()))
        # joblib uses pickle under the hood: only load directories you trust.
        model.pipeline = joblib.load(directory / PIPELINE_FILE)
        return model
