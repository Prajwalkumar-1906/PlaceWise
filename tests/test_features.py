"""Unit tests for feature engineering, preprocessing pipelines, and encoders."""

import numpy as np
import pandas as pd

from src.features.encoders import SalaryBandEncoder, split_features_by_sensitivity
from src.features.pipeline import create_preprocessing_pipeline, get_feature_names


def test_preprocessing_pipeline_fit_transform():
    """Verify ColumnTransformer transforms numerical and categorical columns properly."""
    df = pd.DataFrame(
        {
            "cgpa": [7.5, 8.5, 9.0],
            "tenth_pct": [75.0, 85.0, 90.0],
            "branch": ["Computer Science", "Information Technology", "Computer Science"],
            "gender": ["Male", "Female", "Male"],
        }
    )
    num_cols = ["cgpa", "tenth_pct"]
    cat_cols = ["branch", "gender"]

    preprocessor = create_preprocessing_pipeline(num_cols, cat_cols)
    transformed = preprocessor.fit_transform(df)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 3
    # 2 numerical + 2 branches + 2 genders = 6 transformed columns
    assert transformed.shape[1] == 6


def test_get_feature_names():
    """Verify get_feature_names extracts all encoded column names."""
    df = pd.DataFrame(
        {
            "cgpa": [7.0, 8.0],
            "branch": ["Computer Science", "Civil"],
        }
    )
    num_cols = ["cgpa"]
    cat_cols = ["branch"]

    preprocessor = create_preprocessing_pipeline(num_cols, cat_cols)
    preprocessor.fit(df)

    names = get_feature_names(preprocessor, num_cols, cat_cols)
    assert len(names) == 3
    assert any("cgpa" in n for n in names)
    assert any("Computer Science" in n or "Civil" in n for n in names)


def test_no_data_leakage_pipeline():
    """Verify that scaler parameters are computed strictly on training data."""
    train_df = pd.DataFrame({"cgpa": [6.0, 8.0, 10.0]})  # mean = 8.0
    test_df = pd.DataFrame({"cgpa": [20.0, 30.0]})

    preprocessor = create_preprocessing_pipeline(["cgpa"], [])
    preprocessor.fit(train_df)

    # Inspect scaler mean inside the fitted pipeline
    fitted_mean = preprocessor.named_transformers_["num"].named_steps["scaler"].mean_[0]
    assert np.isclose(fitted_mean, 8.0)

    # Transform test set; verify it does not alter fitted mean
    preprocessor.transform(test_df)
    assert np.isclose(preprocessor.named_transformers_["num"].named_steps["scaler"].mean_[0], 8.0)


def test_salary_band_encoder():
    """Verify SalaryBandEncoder encodes and inverse-transforms correctly."""
    y = pd.Series(["Low", "Mid", "High", "Mid", "Not Placed"])
    encoder = SalaryBandEncoder()
    encoder.fit(y)

    valid_labels = pd.Series(["Low", "High"])
    encoded = encoder.transform(valid_labels)
    assert len(encoded) == 2
    assert isinstance(encoded[0], (int, np.integer))

    decoded = encoder.inverse_transform(encoded)
    assert list(decoded) == ["Low", "High"]


def test_split_features_by_sensitivity():
    """Verify split_features_by_sensitivity partitions features accurately."""
    num_cols = ["cgpa", "aptitude_score"]
    cat_cols = ["branch", "gender", "college_tier"]
    sensitive = ["gender", "college_tier"]

    subsets = split_features_by_sensitivity(num_cols, cat_cols, sensitive)

    assert subsets["aware"]["categorical"] == cat_cols
    assert subsets["unaware"]["categorical"] == ["branch"]
    assert subsets["unaware"]["numerical"] == num_cols
