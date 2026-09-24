from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from framework.contracts.model import Model

type Metric = Callable[[np.ndarray, np.ndarray], float]


def evaluate(
    model: Model,
    features: pd.DataFrame,
    target: np.ndarray,
    metrics: Mapping[str, Metric],
) -> dict[str, float]:
    predictions = model.predict(features)
    return {name: float(metric(target, predictions)) for name, metric in metrics.items()}
