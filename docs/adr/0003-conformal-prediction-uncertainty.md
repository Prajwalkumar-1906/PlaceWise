# ADR 0003: Conformal Prediction, Probability Calibration, and Human-in-the-Loop Abstention

## Status
Accepted

## Date
2026-10-24

## Context
Standard supervised classifiers produce uncalibrated probabilities that frequently overestimate model certainty, particularly for students near historical decision boundaries. Relying solely on raw sigmoid/softmax scores creates severe operational risks:
1. False confidence in high-stakes campus placement decisions.
2. Inability to define formal statistical guarantees for prediction error rates.
3. Lack of an objective protocol for triggering human career counselor intervention.

## Decision
We implemented a three-tier uncertainty estimation framework:
1. **Probability Calibration**:
   - Implemented `ProbabilityCalibrator` supporting both Platt Scaling (logistic sigmoid) and Isotonic Regression.
   - Evaluated calibration quality using Expected Calibration Error (ECE) across 10 uniform reliability bins.
   - Employed `FrozenEstimator` to safely calibrate trained scikit-learn pipelines on dedicated holdout splits without pipeline retraining.
2. **Split Conformal Prediction**:
   - Implemented inductive split conformal prediction with user-specified significance level $\alpha = 0.10$ (guaranteeing $\ge 90\%$ marginal coverage).
   - Generates non-empty prediction sets $C(X) \subseteq \{0, 1\}$:
     - $\{1\}$: High-confidence placement.
     - $\{0\}$: High-confidence non-placement.
     - $\{0, 1\}$: High-uncertainty ambiguity (model cannot statistically distinguish outcomes at the $1-\alpha$ confidence level).
     - $\emptyset$: Distributional outlier / anomaly.
3. **Human-in-the-Loop (HITL) Abstention Zone**:
   - Flag applicants with calibrated placement probability $P(\text{Placement}) \in [0.35, 0.65]$ or conformal set size $\ge 2$.
   - Route these applicants to `"Human-in-the-Loop Review Required"`. The system abstains from automated classification and surfaces the applicant's profile to a career counselor or placement officer with SHAP feature impacts.

## Consequences

### Positive
- Mathematically rigorous coverage guarantee: $P(Y \in C(X)) \ge 1 - \alpha$.
- Reduced operational risk by routing ambiguous edge cases away from automated rejection to human intervention.
- Improved reliability diagram alignment and reduced Brier score loss.

### Trade-offs & Mitigations
- Conformal set $\{0, 1\}$ does not provide a single binary answer; requires downstream systems (FastAPI, Streamlit) to display prediction sets alongside single point estimates.
- Calibration requires reserving a validation split from the training dataset.
