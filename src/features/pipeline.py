"""Preprocessing pipelines for numerical scaling, categorical encoding, and feature extraction."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_preprocessing_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Creates a ColumnTransformer pipeline that scales numerical features and one-hot encodes categoricals.

    Transformers are fit strictly on training data inside downstream modeling pipelines
    to prevent any data leakage.

    Args:
        numerical_features: List of column names to scale.
        categorical_features: List of column names to one-hot encode.

    Returns:
        ColumnTransformer: Unfitted preprocessing pipeline.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.debug(
        f"Created preprocessor with {len(numerical_features)} numerical and "
        f"{len(categorical_features)} categorical features."
    )
    return preprocessor


def get_feature_names(
    preprocessor: ColumnTransformer,
    numerical_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    """Retrieves all transformed feature names from a fitted ColumnTransformer.

    Args:
        preprocessor: Fitted ColumnTransformer.
        numerical_features: Original numerical column names.
        categorical_features: Original categorical column names.

    Returns:
        list[str]: Output feature names after transformation.
    """
    try:
        names = preprocessor.get_feature_names_out()
        return [str(name) for name in names]
    except Exception as exc:
        logger.warning(f"Could not extract feature names directly: {exc}. Reconstructing...")
        feature_names = list(numerical_features)
        try:
            onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = onehot.get_feature_names_out(categorical_features)
            feature_names.extend(list(cat_names))
        except Exception:
            feature_names.extend(categorical_features)
        return feature_names
