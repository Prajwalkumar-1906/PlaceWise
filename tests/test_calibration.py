"""Unit tests for probability calibration, ECE, conformal prediction, and human-in-the-loop abstention."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.calibration import (
    ConformalPredictor,
    ProbabilityCalibrator,
    classify_hitl_confidence,
    compute_ece,
)


def test_compute_ece():
    """Verify Expected Calibration Error computation."""
    # Perfectly calibrated dummy predictions
    y_true = np.array([1] * 90 + [0] * 10)
    y_prob = np.array([0.9] * 100)
    ece_good = compute_ece(y_true, y_prob, n_bins=10)
    assert ece_good == 0.0

    # Miscalibrated predictions
    y_bad_true = np.array([0] * 100)
    y_bad_prob = np.array([0.9] * 100)
    ece_bad = compute_ece(y_bad_true, y_bad_prob, n_bins=10)
    assert ece_bad == 0.9

    # Empty inputs
    assert compute_ece(np.array([]), np.array([])) == 0.0


def test_probability_calibrator_sigmoid_and_isotonic():
    """Verify fitting and probability prediction with Platt scaling and Isotonic regression."""
    np.random.seed(42)
    X = pd.DataFrame({"feat1": np.random.randn(200), "feat2": np.random.randn(200)})
    y = (X["feat1"] + 0.5 * X["feat2"] > 0).astype(int)

    X_train, X_val, X_test = X.iloc[:100], X.iloc[100:150], X.iloc[150:]
    y_train, y_val, y_test = y.iloc[:100], y.iloc[100:150], y.iloc[150:]

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # 1. Sigmoid Calibrator
    cal_sig = ProbabilityCalibrator(clf, method="sigmoid")
    cal_sig.fit(X_val, y_val)
    assert cal_sig.is_fitted
    probs_sig = cal_sig.predict_proba(X_test)
    assert probs_sig.shape == (50, 2)
    assert np.allclose(np.sum(probs_sig, axis=1), 1.0)

    preds_sig = cal_sig.predict(X_test, threshold=0.5)
    assert len(preds_sig) == 50

    eval_sig = cal_sig.evaluate(X_test, y_test)
    assert "uncalibrated" in eval_sig
    assert "calibrated" in eval_sig
    assert "brier_score" in eval_sig["calibrated"]
    assert "ece" in eval_sig["calibrated"]

    # 2. Isotonic Calibrator
    cal_iso = ProbabilityCalibrator(clf, method="isotonic")
    cal_iso.fit(X_val, y_val)
    probs_iso = cal_iso.predict_proba(X_test)
    assert probs_iso.shape == (50, 2)


def test_conformal_predictor():
    """Verify split conformal prediction set generation and coverage guarantee."""
    np.random.seed(42)
    X = pd.DataFrame({"feat1": np.random.randn(300), "feat2": np.random.randn(300)})
    y = (X["feat1"] > 0).astype(int)

    X_train, X_cal, X_test = X.iloc[:100], X.iloc[100:200], X.iloc[200:]
    y_train, y_cal, y_test = y.iloc[:100], y.iloc[100:200], y.iloc[200:]

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    cp = ConformalPredictor(clf, alpha=0.10)
    cp.fit(X_cal, y_cal)
    assert cp.is_fitted
    assert cp.q_hat is not None
    assert 0.0 < cp.q_hat < 1.0

    sets = cp.predict_sets(X_test)
    assert len(sets) == len(X_test)
    for s in sets:
        assert isinstance(s, set)
        assert s.issubset({0, 1})

    summary = cp.predict_summary(X_test)
    assert len(summary) == len(X_test)
    assert "category" in summary[0]
    assert "needs_human_review" in summary[0]

    cov_eval = cp.evaluate_coverage(X_test, y_test)
    assert "empirical_coverage" in cov_eval
    assert "target_coverage" in cov_eval
    assert cov_eval["target_coverage"] == 0.90
    # On synthetic separable data, empirical coverage should satisfy 90% target
    assert cov_eval["empirical_coverage"] >= 0.85


def test_classify_hitl_confidence():
    """Verify human-in-the-loop confidence band classifications."""
    # High confidence placed
    res_high = classify_hitl_confidence(0.85, lower_threshold=0.35, upper_threshold=0.65)
    assert res_high["status"] == "HIGH_CONFIDENCE_PLACED"
    assert res_high["needs_human_review"] is False
    assert res_high["decision"] == "Placed"

    # High confidence unplaced
    res_low = classify_hitl_confidence(0.15, lower_threshold=0.35, upper_threshold=0.65)
    assert res_low["status"] == "HIGH_CONFIDENCE_UNPLACED"
    assert res_low["needs_human_review"] is False
    assert res_low["decision"] == "Not Placed"

    # Borderline / abstention zone
    res_mid = classify_hitl_confidence(0.50, lower_threshold=0.35, upper_threshold=0.65)
    assert res_mid["status"] == "HUMAN_REVIEW_RECOMMENDED"
    assert res_mid["needs_human_review"] is True
    assert "Abstention Zone" in res_mid["confidence_band"]
