from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from zenml import step


@step
def load_data() -> tuple[Annotated[pd.DataFrame, "features"], Annotated[np.ndarray, "target"]]:
    """Load the Breast Cancer Wisconsin dataset as features and labels."""
    # Bundled with scikit-learn: no download needed.
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target.to_numpy()
