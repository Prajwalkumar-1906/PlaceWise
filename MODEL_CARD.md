# Model Card: Campus Placement Predictor & Fairness Auditor

## 1. Model Details
- **Model Name**: Campus Placement Predictor (Binary Classifier) & Compensation Tier Predictor (Multiclass Classifier)
- **Model Version**: 0.1.0-alpha
- **Model Architecture**:
  - Baseline 1: Regularized Logistic Regression (Balanced class weighting)
  - Baseline 2: Random Forest Classifier (Ensemble baseline)
  - Primary Model: Extreme Gradient Boosting (XGBoost) tuned via Optuna Stratified Cross-Validation
- **Frameworks**: scikit-learn, XGBoost, SHAP, Fairlearn, Optuna
- **Date**: October 2026
- **License**: MIT

---

## 2. Intended Use & Scope
- **Intended Use**:
  - Academic counseling and student self-assessment to identify high-leverage skill improvements (e.g. DSA problem count, internship experience).
  - Algorithmic fairness benchmarking and bias audit demonstration.
- **Out-of-Scope & Unacceptable Use**:
  - Automated screening or rejection of job applicants.
  - Institutional rank filtering without human-in-the-loop review.
  - Deciding scholarships, admissions, or compensation bands in production settings.

---

## 3. Training & Validation Setup
- **Dataset**: Stratified 70% Train, 15% Validation, 15% Held-Out Test split.
- **Preprocessing Pipeline**:
  - Standard scaling on numerical features (fit strictly on training data).
  - One-hot encoding on categorical features with handle_unknown='ignore'.
  - Target-agnostic feature transformation preventing any data leakage.
- **Evaluation Metrics**:
  - Primary: ROC-AUC, PR-AUC
  - Secondary: Accuracy, Precision, Recall, Macro-F1, Calibration (Brier Score)
  - Fairness: Demographic Parity Difference/Ratio, Equalized Odds Difference, Equal Opportunity Gap

---

## 4. Explainability & Interpretability
- **Global Explanations**: SHAP TreeExplainer beeswarm plots, mean absolute SHAP feature importance, and partial dependence interactions.
- **Local Explanations**: Individual student waterfall plots with exact log-odds additivity checks:
  $$\text{Model Output} = \text{Base Value} + \sum_{i=1}^{M} \phi_i$$
  and calibrated conversion to probability-point contribution approximations.
- **Counterfactual "What-If" Analysis**: Real-time simulation of actionable feature modifications (e.g., +100 DSA problems solved, +1 internship).

---

## 5. Fairness Audit & Bias Mitigation Matrix
The model is audited against sensitive demographic attributes:
- `gender` (Male vs. Female)
- `college_tier` (Tier 1 vs. Tier 2 vs. Tier 3)
- Intersectional: `gender` $\times$ `college_tier`

### Mitigation Algorithms Evaluated:
1. **Pre-processing**: Fairlearn Sample Reweighing (adjusting class and group weights prior to training).
2. **Post-processing**: Fairlearn `ThresholdOptimizer` (calibrating group-specific classification thresholds to enforce Equalized Odds or Demographic Parity).
3. **In-processing**: Fairlearn `ExponentiatedGradient` with Demographic Parity / Equalized Odds constraints.

*(Note: Exact quantitative before/after metric tables will be populated upon completion of Phase 2 and Phase 4 execution.)*

---

## 6. Caveats & Ethical Considerations
- Fairness metrics often exist in mathematical tension (e.g., individual fairness vs. group parity).
- Blind removal of sensitive attributes ("fairness through unawareness") does not eliminate bias due to proxy feature correlations.
- Mitigated models require explicit trade-off justification between predictive utility and group parity.
