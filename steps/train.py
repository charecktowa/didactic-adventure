import importlib
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import step

from framework.contracts.model import Model
from framework.validation import validate_model

VALIDATION_ROWS = 100


@step
def train(model_name: str, x_train: pd.DataFrame, y_train: np.ndarray) -> Annotated[Model, "model"]:
    """Build the model in `models/<model_name>`, validate it on a sample, then fit it."""
    model = importlib.import_module(f"models.{model_name}.model").build_model()
    # Fail fast on a small sample before spending time on the real training.
    validate_model(model, x_train.iloc[:VALIDATION_ROWS], y_train[:VALIDATION_ROWS], trainable=True)
    return model.fit(x_train, y_train)
