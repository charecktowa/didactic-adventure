from typing import Annotated

import numpy as np
import pandas as pd
from zenml import step

import models
from framework.contracts.model import Model, Trainable
from framework.validation import validate_model
from materializers.model import ModelMaterializer

VALIDATION_ROWS = 100


@step(output_materializers=ModelMaterializer)
def train(model_name: str, x_train: pd.DataFrame, y_train: np.ndarray) -> Annotated[Model, "model"]:
    """Build the model in `models/<model_name>`, validate it on a sample, then fit it."""
    model = models.build(model_name)
    # Fail fast on a small sample before spending time on the real training.
    validate_model(model, x_train.iloc[:VALIDATION_ROWS], y_train[:VALIDATION_ROWS], trainable=True)
    if not isinstance(model, Trainable):  # Never true after validate_model(trainable=True).
        raise TypeError(f"{model_name} cannot be trained: it has no fit()")
    return model.fit(x_train, y_train)
