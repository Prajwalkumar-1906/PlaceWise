"""Bias mitigation algorithms: Pre-processing (Reweighing), In-processing (ExponentiatedGradient), and Post-processing (ThresholdOptimizer)."""

from typing import Any

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.pipeline import create_preprocessing_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_sample_weights(
    y: pd.Series | np.ndarray,
    sensitive_features: pd.Series | np.ndarray,
) -> np.ndarray:
    """Computes demographic parity reweighing sample weights.

    Formula:
        W(y, s) = (P(Y=y) * P(S=s)) / P(Y=y, S=s)

    Args:
        y: Binary ground truth labels.
        sensitive_features: Sensitive attribute series.

    Returns:
        np.ndarray: Per-sample weights array normalized to mean 1.0.
    """
    y_arr = np.array(y).astype(int)
    s_arr = pd.Series(sensitive_features).astype(str).values
    n = len(y_arr)

    weights = np.ones(n, dtype=float)

    # Calculate marginal and joint probabilities
    p_y1 = np.mean(y_arr == 1)
    p_y0 = 1.0 - p_y1

    unique_groups = np.unique(s_arr)

    for g in unique_groups:
        mask_g = s_arr == g
        p_g = np.mean(mask_g)

        # Joint: Y=1, S=g
        mask_y1_g = (y_arr == 1) & mask_g
        p_y1_g = np.mean(mask_y1_g)

        # Joint: Y=0, S=g
        mask_y0_g = (y_arr == 0) & mask_g
        p_y0_g = np.mean(mask_y0_g)

        if p_y1_g > 0:
            weights[mask_y1_g] = (p_y1 * p_g) / p_y1_g
        if p_y0_g > 0:
            weights[mask_y0_g] = (p_y0 * p_g) / p_y0_g

    # Normalize weights
    weights = weights / np.mean(weights)
    return weights


def apply_reweighing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_train: pd.Series,
    numerical_features: list[str],
    categorical_features: list[str],
    xgb_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Trains an XGBoost model using pre-processing sample reweighing.

    Args:
        X_train: Training features.
        y_train: Training labels.
        sensitive_train: Training sensitive attribute.
        numerical_features: Numerical column names.
        categorical_features: Categorical column names.
        xgb_params: Optional hyperparameters for XGBoost.

    Returns:
        Pipeline: Fitted reweighed model pipeline.
    """
    weights = compute_sample_weights(y_train, sensitive_train)

    preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)
    X_proc = preprocessor.fit_transform(X_train)

    params = xgb_params or {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "eval_metric": "logloss",
    }
    clf = XGBClassifier(**params)
    clf.fit(X_proc, y_train, sample_weight=weights)

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    logger.info("Successfully trained Pre-processing Reweighted XGBoost model.")
    return pipeline


class Float64EstimatorWrapper(ClassifierMixin, BaseEstimator):
    """Wraps an estimator to ensure predict_proba returns float64, preventing pandas 3.0 downcast errors."""

    def __init__(self, estimator: Any = None) -> None:
        self.estimator = estimator
        if estimator is not None:
            self.classes_ = np.asarray(getattr(estimator, "classes_", [0, 1]))
            self.is_fitted_ = True

    def fit(self, X: Any, y: Any = None) -> "Float64EstimatorWrapper":
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X), dtype=np.float64)

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=int)


def fit_threshold_optimizer(
    unmitigated_pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sensitive_val: pd.Series,
    constraint: str = "demographic_parity",
) -> ThresholdOptimizer:
    """Fits a post-processing Fairlearn ThresholdOptimizer on validation data.

    Tunes group-specific decision thresholds to satisfy demographic parity or equalized odds.

    Args:
        unmitigated_pipeline: Pre-trained baseline pipeline.
        X_val: Validation features.
        y_val: Validation labels.
        sensitive_val: Validation sensitive attributes.
        constraint: 'demographic_parity' or 'equalized_odds'.

    Returns:
        ThresholdOptimizer: Fitted post-processor.
    """
    wrapper = Float64EstimatorWrapper(unmitigated_pipeline)
    optimizer = ThresholdOptimizer(
        estimator=wrapper,
        constraints=constraint,
        predict_method="predict_proba",
        prefit=True,
    )
    optimizer.fit(X_val, y_val, sensitive_features=np.asarray(sensitive_val, dtype=str))
    logger.info(f"Fitted ThresholdOptimizer post-processor with constraint='{constraint}'.")
    return optimizer


def fit_exponentiated_gradient(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_train: pd.Series,
    numerical_features: list[str],
    categorical_features: list[str],
    eps: float = 0.02,
) -> tuple[Pipeline, ExponentiatedGradient]:
    """Fits an in-processing Fairlearn ExponentiatedGradient reductions model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        sensitive_train: Training sensitive attributes.
        numerical_features: Numerical columns.
        categorical_features: Categorical columns.
        eps: Desired bound on constraint violation.

    Returns:
        tuple (fitted_preprocessor_pipeline, fitted_exp_grad_model)
    """
    preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)
    X_proc = preprocessor.fit_transform(X_train)

    base_estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
    exp_grad = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=DemographicParity(),
        eps=eps,
        max_iter=25,
    )
    exp_grad.fit(X_proc, y_train, sensitive_features=sensitive_train)
    logger.info("Fitted ExponentiatedGradient reductions in-processing model.")
    return preprocessor, exp_grad


class MitigationManager:
    """Coordinates training, post-processing, and inference across all mitigation strategies."""

    def __init__(
        self,
        unmitigated_pipeline: Pipeline,
        numerical_features: list[str],
        categorical_features: list[str],
    ) -> None:
        self.unmitigated_pipeline = unmitigated_pipeline
        self.num_features = numerical_features
        self.cat_features = categorical_features

        self.reweighted_pipeline: Pipeline | None = None
        self.threshold_optimizer: ThresholdOptimizer | None = None
        self.exp_grad_preprocessor: Pipeline | None = None
        self.exp_grad_model: ExponentiatedGradient | None = None

    def fit_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        sensitive_val: pd.Series,
    ) -> None:
        """Trains and fits all three mitigation methods."""
        # 1. Pre-processing: Reweighing
        self.reweighted_pipeline = apply_reweighing(
            X_train=X_train,
            y_train=y_train,
            sensitive_train=sensitive_train,
            numerical_features=self.num_features,
            categorical_features=self.cat_features,
        )

        # 2. Post-processing: ThresholdOptimizer
        self.threshold_optimizer = fit_threshold_optimizer(
            unmitigated_pipeline=self.unmitigated_pipeline,
            X_val=X_val,
            y_val=y_val,
            sensitive_val=sensitive_val,
            constraint="demographic_parity",
        )

        # 3. In-processing: ExponentiatedGradient
        self.exp_grad_preprocessor, self.exp_grad_model = fit_exponentiated_gradient(
            X_train=X_train,
            y_train=y_train,
            sensitive_train=sensitive_train,
            numerical_features=self.num_features,
            categorical_features=self.cat_features,
        )

    def predict_reweighted(self, X: pd.DataFrame) -> np.ndarray:
        """Predictions from pre-processing reweighted model."""
        if self.reweighted_pipeline is None:
            raise RuntimeError("Reweighted pipeline is not fitted.")
        return self.reweighted_pipeline.predict(X)

    def predict_threshold_optimizer(
        self, X: pd.DataFrame, sensitive_features: pd.Series
    ) -> np.ndarray:
        """Predictions from post-processing ThresholdOptimizer."""
        if self.threshold_optimizer is None:
            raise RuntimeError("ThresholdOptimizer is not fitted.")
        sens_arr = np.asarray(sensitive_features, dtype=str)
        preds = self.threshold_optimizer.predict(X, sensitive_features=sens_arr)
        return np.asarray(preds, dtype=int)

    def predict_exponentiated_gradient(self, X: pd.DataFrame) -> np.ndarray:
        """Predictions from in-processing ExponentiatedGradient."""
        if self.exp_grad_preprocessor is None or self.exp_grad_model is None:
            raise RuntimeError("ExponentiatedGradient is not fitted.")
        X_proc = self.exp_grad_preprocessor.transform(X)
        return self.exp_grad_model.predict(X_proc)
