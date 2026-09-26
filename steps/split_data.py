from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step


@step
def split_data(
    features: pd.DataFrame, target: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
    """Split the data into stratified train and test sets."""
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, stratify=target, random_state=random_state
    )
    return x_train, x_test, y_train, y_test
