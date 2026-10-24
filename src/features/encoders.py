"""Feature transformation helpers and target label encoding utilities."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SalaryBandEncoder:
    """Encoder for multiclass compensation targets ('Low', 'Mid', 'High')."""

    ORDERED_CLASSES = ["Low", "Mid", "High"]

    def __init__(self) -> None:
        self.encoder = LabelEncoder()
        self.classes_: list[str] = []

    def fit(self, y: pd.Series | np.ndarray) -> "SalaryBandEncoder":
        """Fits the label encoder on non-null placement salary bands."""
        filtered_y = pd.Series(y).dropna()
        filtered_y = filtered_y[filtered_y != "Not Placed"]
        self.encoder.fit(filtered_y)
        self.classes_ = list(self.encoder.classes_)
        logger.debug(f"Fitted SalaryBandEncoder with classes: {self.classes_}")
        return self

    def transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        """Transforms salary band string labels to integer classes."""
        return self.encoder.transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Inverts integer predictions back to readable salary bands."""
        return self.encoder.inverse_transform(y)


def split_features_by_sensitivity(
    numerical_features: list[str],
    categorical_features: list[str],
    sensitive_features: list[str],
) -> dict[str, dict[str, list[str]]]:
    """Splits feature definitions into standard and sensitive-unaware subsets.

    Used in Phase 4 fairness audit to train models with 'fairness through unawareness'.

    Args:
        numerical_features: List of numerical columns.
        categorical_features: List of categorical columns.
        sensitive_features: List of sensitive attribute columns to exclude.

    Returns:
        dict containing 'aware' and 'unaware' feature definitions.
    """
    unaware_num = [f for f in numerical_features if f not in sensitive_features]
    unaware_cat = [f for f in categorical_features if f not in sensitive_features]

    return {
        "aware": {
            "numerical": list(numerical_features),
            "categorical": list(categorical_features),
        },
        "unaware": {
            "numerical": unaware_num,
            "categorical": unaware_cat,
        },
    }
