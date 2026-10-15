# Model Card: Campus Placement Predictor & Fairness Auditor

## 1. Model Details
- **Model Name**: Campus Placement Predictor (Binary Classifier) & Compensation Tier Predictor (Multiclass Classifier)
- **Model Version**: 0.1.0 (Production Release)
- **Model Architecture**:
  - Baseline 1: Regularized Logistic Regression (Balanced class weighting)
  - Baseline 2: Random Forest Classifier (Ensemble bagging baseline)
  - Primary Model: Extreme Gradient Boosting (XGBoost) tuned via Optuna 5-fold Stratified Cross-Validation
  - Salary Model: Gradient Boosting 3-Class Classifier (Low <4 LPA, Mid 4-8 LPA, High >8 LPA)
- **Frameworks**: Python 3.14+, scikit-learn 1.7+, XGBoost 3.1+, Fairlearn 0.13+, SHAP 0.51+, Optuna 4.7+, Streamlit 1.66+
- **Date**: October 2026
- **License**: MIT

---

## 2. Intended Use & Scope
- **Intended Use**:
  - Academic counseling and student self-assessment to identify high-leverage skill improvements (e.g. DSA problem count, internship experience).
  - Algorithmic fairness benchmarking and bias audit demonstration.
  - Interactive decision support with real-time what-if counterfactual levers.
- **Out-of-Scope & Unacceptable Use**:
  - Automated screening or rejection of job applicants without human recruiter review.
  - Institutional rank filtering without holistic evaluation.
  - Deciding scholarships, admissions, or compensation bands in binding production settings.

---

## 3. Quantitative Performance Benchmark (Held-Out Test Data, N=750)

### Primary Placement Model Benchmark
| Model Architecture | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **89.60%** | 91.69% | **96.77%** | **0.9416** | 0.8735 | 0.9762 | **0.0773** |
| Random Forest | 85.33% | 94.41% | 88.31% | 0.9126 | 0.8755 | 0.9762 | 0.1219 |
| Logistic Regression | 81.87% | **96.22%** | 82.31% | 0.8872 | **0.8926** | **0.9801** | 0.1256 |

*All metrics loaded from `reports/metrics.json`.*

### Multiclass Salary Compensation Model (Placed Candidates, N=650)
- **Test Accuracy**: 84.46%
- **Macro F1-Score**: 0.8467
- **Weighted F1-Score**: 0.8450
- **Class Breakdown**:
  - High (>8 LPA): Precision = 88.57%, Recall = 82.67%, F1 = 0.8552
  - Mid (4-8 LPA): Precision = 80.28%, Recall = 83.82%, F1 = 0.8201
  - Low (<4 LPA): Precision = 87.50%, Recall = 85.48%, F1 = 0.8648

---

## 4. Explainability & Interpretability
- **Global Explanations**: SHAP `TreeExplainer` beeswarm plots (`reports/figures/shap_beeswarm.png`) and mean absolute SHAP feature importance rankings.
- **Local Explanations**: Individual student waterfall plots with exact log-odds additivity checks:
  $$\text{Model Margin (log-odds)} = \text{Base Value} + \sum_{i=1}^{M} \phi_i$$
  where numerical error is empirically verified ($|\text{error}| < 10^{-5}$).
- **Counterfactual "What-If" Analysis**: Real-time simulation of actionable feature modifications (e.g. +150 DSA problems, +1 internship, clearing backlogs).

---

## 5. Fairness Audit & Bias Mitigation Matrix (College Tier)

Benchmarked on ground-truth institutional bias data (where Tier 3 candidates face systematic penalties at equal skill levels):

| Model Architecture | Test Accuracy | Demographic Parity Gap | DP Ratio | Equalized Odds Gap | Bias Reduction % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Aware)** | 80.80% | 0.0708 | 0.9002 | 0.0485 | **+0.0%** |
| **Unawareness (Blind)** | 84.27% | 0.0197 | 0.9772 | 0.0913 | **+72.2%** |
| **Mitigated (Reweighing)** | **85.47%** | **0.0268** | **0.9692** | **0.0408** | **+62.1%** |
| **Mitigated (ThresholdOptimizer)** | 84.80% | 0.0784 | 0.9061 | 0.0833 | **-10.6%** |
| **Mitigated (ExponentiatedGradient)** | 80.00% | 0.0667 | 0.9042 | 0.0894 | **+5.9%** |

$$\text{Bias Reduction \%} = \frac{\text{DP\_Diff}_{\text{baseline}} - \text{DP\_Diff}_{\text{mitigated}}}{\text{DP\_Diff}_{\text{baseline}}} \times 100\%$$

### Production Recommendation
- **Pre-processing Reweighing** achieves the most balanced Pareto trade-off: reducing demographic parity gap from 7.1% to 2.7% (62.1% bias reduction) while improving test accuracy to 85.5% without requiring discriminatory group-specific run-time thresholds.

---

## 6. Caveats & Ethical Considerations
- **Failure of Unawareness**: Removing `gender` and `college_tier` does not prevent algorithmic bias due to proxy correlations in aptitude, coding volume, and branch.
- **Mathematical Incompatibility**: Demographic Parity, Equalized Odds, and Calibration cannot simultaneously hold when underlying base rates differ (Chouldechova, 2017; Kleinberg et al., 2016).
- **Human-in-the-Loop Oversight**: In compliance with the EU AI Act (High-Risk AI Systems) and EEOC 80% Rule, automated decisions must always be audited and supplemented by human counselors.
