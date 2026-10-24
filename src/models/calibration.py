"""Uncertainty quantification, probability calibration, conformal prediction, and human-in-the-loop abstention."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

try:
    from sklearn.frozen import FrozenEstimator

    HAS_FROZEN = True
except ImportError:
    HAS_FROZEN = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_ece(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series, n_bins: int = 10
) -> float:
    """Computes Expected Calibration Error (ECE) across equal-width probability bins.

    Formula:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted probabilities for positive class in [0, 1].
        n_bins: Number of confidence bins (default 10).

    Returns:
        float: Expected calibration error in [0, 1].
    """
    y_t = np.array(y_true).astype(int)
    y_p = np.array(y_prob).astype(float)
    n = len(y_t)
    if n == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        # Include right edge for the last bin
        if i == n_bins - 1:
            mask = (y_p >= low) & (y_p <= high)
        else:
            mask = (y_p >= low) & (y_p < high)

        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_t[mask])
            bin_conf = np.mean(y_p[mask])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)

    return round(float(ece), 4)


class ProbabilityCalibrator:
    """Calibrates classifier probabilities using Platt scaling (Sigmoid) or Isotonic regression strictly on validation splits."""

    def __init__(self, base_pipeline: Any, method: str = "sigmoid") -> None:
        """Initializes the calibrator.

        Args:
            base_pipeline: Trained scikit-learn Pipeline or estimator.
            method: Calibration method ('sigmoid' for Platt scaling, 'isotonic' for non-parametric).
        """
        self.base_pipeline = base_pipeline
        self.method = method
        self.calibrated_classifier: Any = None
        self.is_fitted = False

    def fit(
        self, X_val: pd.DataFrame | np.ndarray, y_val: pd.Series | np.ndarray
    ) -> "ProbabilityCalibrator":
        """Fits calibration parameters strictly on held-out validation data without refitting base pipeline.

        Args:
            X_val: Validation feature matrix.
            y_val: Validation true binary labels.

        Returns:
            self: Fitted ProbabilityCalibrator instance.
        """
        logger.info(
            f"Fitting probability calibrator using method='{self.method}' on validation split ({len(y_val)} samples)..."
        )

        if HAS_FROZEN:
            estimator_to_use = FrozenEstimator(self.base_pipeline)
            self.calibrated_classifier = CalibratedClassifierCV(
                estimator=estimator_to_use,
                method=self.method,
            )
        else:
            self.calibrated_classifier = CalibratedClassifierCV(
                estimator=self.base_pipeline,
                method=self.method,
                cv="prefit",
            )

        self.calibrated_classifier.fit(X_val, y_val)
        self.is_fitted = True
        logger.info("Calibration fitting complete.")
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns calibrated probability estimates [P(Y=0), P(Y=1)]."""
        if not self.is_fitted or self.calibrated_classifier is None:
            return self.base_pipeline.predict_proba(X)
        return self.calibrated_classifier.predict_proba(X)

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions based on calibrated probability threshold."""
        prob_pos = self.predict_proba(X)[:, 1]
        return (prob_pos >= threshold).astype(int)

    def evaluate(
        self, X_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray
    ) -> dict[str, Any]:
        """Compares Brier score and ECE before and after calibration.

        Args:
            X_test: Test feature matrix.
            y_test: Test true binary labels.

        Returns:
            dict containing uncalibrated vs calibrated Brier scores and ECEs.
        """
        y_t = np.array(y_test).astype(int)

        # Uncalibrated metrics
        uncal_probs = self.base_pipeline.predict_proba(X_test)[:, 1]
        uncal_brier = float(brier_score_loss(y_t, uncal_probs))
        uncal_ece = compute_ece(y_t, uncal_probs)

        # Calibrated metrics
        cal_probs = self.predict_proba(X_test)[:, 1]
        cal_brier = float(brier_score_loss(y_t, cal_probs))
        cal_ece = compute_ece(y_t, cal_probs)

        return {
            "method": self.method,
            "uncalibrated": {
                "brier_score": round(uncal_brier, 4),
                "ece": uncal_ece,
            },
            "calibrated": {
                "brier_score": round(cal_brier, 4),
                "ece": cal_ece,
            },
            "brier_reduction_pct": (
                round(float((uncal_brier - cal_brier) / uncal_brier * 100.0), 2)
                if uncal_brier > 0
                else 0.0
            ),
            "ece_reduction_pct": (
                round(float((uncal_ece - cal_ece) / uncal_ece * 100.0), 2) if uncal_ece > 0 else 0.0
            ),
        }


class ConformalPredictor:
    """Split Conformal Predictor for binary classification providing finite-sample coverage guarantees."""

    def __init__(self, pipeline_or_calibrator: Any, alpha: float = 0.10) -> None:
        """Initializes Conformal Predictor.

        Args:
            pipeline_or_calibrator: Fitted pipeline or ProbabilityCalibrator supporting predict_proba.
            alpha: Significance level (default 0.10 for 90% confidence guarantee: Coverage >= 1 - alpha).
        """
        self.predictor = pipeline_or_calibrator
        self.alpha = alpha
        self.q_hat: float | None = None
        self.is_fitted = False

    def fit(
        self, X_cal: pd.DataFrame | np.ndarray, y_cal: pd.Series | np.ndarray
    ) -> "ConformalPredictor":
        """Calculates non-conformity score quantile on calibration set (e.g. validation split).

        Non-conformity score:
            s_i = 1 - P(Y = y_i | X_i)

        Args:
            X_cal: Calibration feature matrix.
            y_cal: Calibration binary true labels (0 or 1).

        Returns:
            self: Fitted ConformalPredictor instance.
        """
        y_c = np.array(y_cal).astype(int)
        probs = self.predictor.predict_proba(X_cal)[:, 1]

        # Score for true class: if y=1 -> 1-p, if y=0 -> p
        scores = np.where(y_c == 1, 1.0 - probs, probs)
        n = len(scores)

        # Conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - self.alpha)) / n))
        self.q_hat = float(
            np.quantile(scores, q_level, method="higher" if hasattr(np, "quantile") else "linear")
        )
        self.is_fitted = True

        logger.info(
            f"Fitted Conformal Predictor (alpha={self.alpha:.2f}, n_cal={n}): Conformal quantile q_hat = {self.q_hat:.4f}"
        )
        return self

    def predict_sets(self, X: pd.DataFrame | np.ndarray) -> list[set[int]]:
        """Generates conformal prediction sets for applicants in X.

        For candidate with P(placed) = p:
            1 in set if (1 - p) <= q_hat  <=> p >= 1 - q_hat
            0 in set if p <= q_hat

        Returns:
            list of sets, each containing {1}, {0}, {0, 1}, or empty set ().
        """
        if not self.is_fitted or self.q_hat is None:
            raise ValueError(
                "ConformalPredictor must be fitted on calibration data before predict_sets()."
            )

        probs = self.predictor.predict_proba(X)[:, 1]
        prediction_sets: list[set[int]] = []

        for p in probs:
            s = set()
            # Check class 1
            if (1.0 - p) <= self.q_hat:
                s.add(1)
            # Check class 0
            if p <= self.q_hat:
                s.add(0)
            prediction_sets.append(s)

        return prediction_sets

    def predict_summary(self, X: pd.DataFrame | np.ndarray) -> list[dict[str, Any]]:
        """Returns structured prediction set summaries with classification and human-in-the-loop tags."""
        sets = self.predict_sets(X)
        probs = self.predictor.predict_proba(X)[:, 1]
        summaries = []

        for p, s in zip(probs, sets):
            if s == {1}:
                category = "Confident Placed"
                label = "Placed"
                needs_review = False
            elif s == {0}:
                category = "Confident Unplaced"
                label = "Not Placed"
                needs_review = False
            elif s == {0, 1}:
                category = "Uncertain / Both"
                label = "Borderline / Abstain"
                needs_review = True
            else:
                category = "Out of Distribution / Empty"
                label = "Anomaly"
                needs_review = True

            summaries.append(
                {
                    "placement_probability": round(float(p), 4),
                    "prediction_set": sorted(s),
                    "set_size": len(s),
                    "category": category,
                    "label": label,
                    "needs_human_review": needs_review,
                }
            )

        return summaries

    def evaluate_coverage(
        self, X_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray
    ) -> dict[str, Any]:
        """Evaluates empirical coverage and set size distributions on held-out test data."""
        y_t = np.array(y_test).astype(int)
        sets = self.predict_sets(X_test)
        n = len(y_t)

        covered = sum(1 for yt, s in zip(y_t, sets) if yt in s)
        empirical_coverage = float(covered / n) if n > 0 else 0.0

        singleton_count = sum(1 for s in sets if len(s) == 1)
        both_count = sum(1 for s in sets if len(s) == 2)
        empty_count = sum(1 for s in sets if len(s) == 0)

        return {
            "significance_level_alpha": self.alpha,
            "target_coverage": round(1.0 - self.alpha, 4),
            "empirical_coverage": round(empirical_coverage, 4),
            "guarantee_satisfied": empirical_coverage >= (1.0 - self.alpha),
            "conformal_quantile_q_hat": round(self.q_hat, 4) if self.q_hat else 0.0,
            "average_set_size": round(float(np.mean([len(s) for s in sets])), 3),
            "singleton_pct": round(float(singleton_count / n * 100.0), 2),
            "uncertain_both_pct": round(float(both_count / n * 100.0), 2),
            "empty_pct": round(float(empty_count / n * 100.0), 2),
        }


def classify_hitl_confidence(
    probability: float,
    lower_threshold: float = 0.35,
    upper_threshold: float = 0.65,
) -> dict[str, Any]:
    """Classifies an applicant's prediction into High Confidence vs Human Review Abstention Zone.

    Args:
        probability: Calibrated placement probability in [0, 1].
        lower_threshold: Lower bound for human review (below this is confident unplaced).
        upper_threshold: Upper bound for human review (above this is confident placed).

    Returns:
        dict with status, tag, confidence_level, and guidance recommendation.
    """
    p = float(probability)
    if p > upper_threshold:
        return {
            "probability": round(p, 4),
            "decision": "Placed",
            "confidence_band": "High Confidence",
            "status": "HIGH_CONFIDENCE_PLACED",
            "needs_human_review": False,
            "recommendation": "Applicant exhibits strong placement drivers across academic and technical competencies. Standard placement pipeline.",
        }
    elif p < lower_threshold:
        return {
            "probability": round(p, 4),
            "decision": "Not Placed",
            "confidence_band": "High Confidence",
            "status": "HIGH_CONFIDENCE_UNPLACED",
            "needs_human_review": False,
            "recommendation": "Applicant exhibits significant risk factors. Academic counseling and targeted skill building recommended.",
        }
    else:
        return {
            "probability": round(p, 4),
            "decision": "Uncertain / Borderline",
            "confidence_band": "Abstention Zone (Human Review)",
            "status": "HUMAN_REVIEW_RECOMMENDED",
            "needs_human_review": True,
            "recommendation": f"Placement probability ({p*100:.1f}%) lies in the Model Abstention Zone [{lower_threshold*100:.0f}%, {upper_threshold*100:.0f}%]. Manual placement officer review or interview assessment recommended prior to final evaluation.",
        }
