from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from zenml import step

from framework.contracts.model import Model
from framework.evaluation import evaluate as evaluate_model

METRICS = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}


@step
def evaluate(
    model: Model, x_test: pd.DataFrame, y_test: np.ndarray
) -> Annotated[dict[str, float], "metrics"]:
    """Score the trained model on the held-out test set."""
    return evaluate_model(model, x_test, y_test, METRICS)
