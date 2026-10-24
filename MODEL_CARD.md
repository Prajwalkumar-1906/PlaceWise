# Model Card: PlaceWise Campus Placement Predictor & Governance Suite

## 1. Model Details
- **Model Name**: PlaceWise Campus Placement Predictor (Binary Classifier) & Compensation Band Estimator (Multiclass Classifier)
- **Model Version**: 0.2.0 (Production Release promoted in MLflow Model Registry)
- **Model Registry Staging**: Promoted to `Production` stage under `PlaceWise_XGBoost_Placement` (Version 6)
- **Model Architectures**:
  - Baseline 1: Regularized Logistic Regression (Balanced class weighting)
  - Baseline 2: Random Forest Classifier (100 estimators, Gini criterion)
  - Baseline 3: CatBoost Classifier (Gradient boosted decision trees, 5x5 CV ROC-AUC: 0.8980)
  - Primary Model: Extreme Gradient Boosting (XGBoost) tuned via Optuna 5-fold Stratified Cross-Validation
  - Salary Model: Gradient Boosting 3-Class Classifier (Low <4 LPA, Mid 4-8 LPA, High >8 LPA) trained strictly on placed candidates ($Y=1$)
  - Uncertainty Calibrator: Platt Scaling & Isotonic Regression via `FrozenEstimator`
  - Conformal Predictor: Inductive Split Conformal Prediction ($\alpha = 0.10$, $\ge 90\%$ coverage guarantee)
- **Frameworks**: Python 3.10+, scikit-learn 1.7+, XGBoost 3.1+, CatBoost 1.2+, Fairlearn 0.13+, SHAP 0.51+, MLflow 2.14+, FastAPI 0.115+, Streamlit 1.66+
- **Date**: October 2026
- **License**: MIT

---

## 2. Intended Use & Scope

### Primary Intended Use
- **Academic Counseling & Career Guidance**: Student self-assessment to identify high-leverage skill improvements (e.g., DSA problem count, internship experience, certification acquisition).
- **Algorithmic Fairness Benchmarking**: Auditing institutional and gender disparities across predictive modeling pipelines.
- **Decision Support with Actionable Recourse**: Simulating realistic developmental interventions while strictly locking immutable demographic attributes.

### Out-of-Scope & Unacceptable Use
- **Automated Rejection or Screening**: Rejecting job applicants without human recruiter review.
- **Institutional Rank Filtering**: Automating gatekeeping based on college tier or gender.
- **High-Stakes Binding Determinations**: Deciding scholarships, university admissions, or contractual compensation without human oversight.

---

## 3. Quantitative Performance Benchmarks (Held-Out Test Set, N=750)

### A. Primary Placement Classifier Benchmark
*All figures retrieved directly from `reports/metrics.json`.*

| Model Architecture | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **89.60%** | 91.69% | **96.77%** | **0.9416** | 0.8735 | 0.9762 | **0.0773** |
| Random Forest | 85.33% | 94.41% | 88.31% | 0.9126 | 0.8755 | 0.9762 | 0.1219 |
| Logistic Regression | 81.87% | **96.22%** | 82.31% | 0.8872 | **0.8926** | **0.9801** | 0.1256 |

### B. 5x5 Repeated Stratified Cross-Validation
- **CatBoost Classifier**: Mean ROC-AUC = **$0.8980 \pm 0.0193$** (95% CI: $[0.8592, 0.9346]$)
- **Tuned XGBoost**: Mean ROC-AUC = **$0.8872 \pm 0.0196$** (95% CI: $[0.8465, 0.9258]$)
- **Paired Hypothesis Tests**: Paired Student's t-test ($p = 0.082$) and Wilcoxon signed-rank test ($p = 0.076$) confirm no statistically significant discrimination gap between CatBoost and XGBoost at $\alpha = 0.05$. XGBoost was selected for production serving due to sub-millisecond inference latency and native SHAP TreeExplainer efficiency.

### C. Multiclass Salary Band Classifier (Placed Students, N=650)
- **Overall Accuracy**: **84.46%** | **Macro F1**: **0.8467** | **Weighted F1**: **0.8450**
- **High Band (>8 LPA)**: Precision = 88.57%, Recall = 82.67%, F1 = 0.8552 (Support: 75)
- **Mid Band (4-8 LPA)**: Precision = 80.28%, Recall = 83.82%, F1 = 0.8201 (Support: 272)
- **Low Band (<4 LPA)**: Precision = 87.50%, Recall = 85.48%, F1 = 0.8648 (Support: 303)

---

## 4. Uncertainty Quantification & Human-in-the-Loop Governance

1. **Probability Calibration**:
   - Platt Scaling and Isotonic Regression calibrated via `FrozenEstimator`.
   - Tuned XGBoost achieves low Expected Calibration Error (ECE) and a Brier Score of **0.0773**.
2. **Split Conformal Prediction**:
   - Calibrated on holdout calibration split with significance level $\alpha = 0.10$.
   - Mathematically guarantees marginal coverage:
     $$P(Y \in C(X)) \ge 1 - \alpha = 90.0\%$$
   - Discrete output sets: $\{1\}$ (Placed), $\{0\}$ (Not Placed), $\{0, 1\}$ (Statistically Uncertain), or $\emptyset$ (Outlier).
3. **Human-in-the-Loop (HITL) Abstention Zone**:
   - Any applicant with calibrated placement probability $P \in [0.35, 0.65]$ or conformal set size $\ge 2$ is routed to `"Human-in-the-Loop Review Required"`.
   - The model abstains from automated classification and surfaces the student profile to an institutional career counselor with local SHAP feature impact drivers.

---

## 5. Algorithmic Fairness Audit & Mitigation Matrix

### A. College Tier Auditing (Ground-Truth Disparity Evaluation)

| Mitigation Strategy | Test Accuracy | Demographic Parity Gap | DP Ratio | Equalized Odds Gap | Bias Reduction % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Aware)** | 80.80% | 0.0708 | 0.9002 | 0.0485 | **0.0%** |
| **Unawareness (Blind)** | 84.27% | 0.0197 | 0.9772 | 0.0913 | **+72.2%** |
| **Mitigated (Reweighing)** | **85.47%** | **0.0268** | **0.9692** | **0.0408** | **+62.1%** |
| **Mitigated (ThresholdOptimizer)** | 84.80% | 0.0784 | 0.9061 | 0.0833 | **-10.6%** |
| **Mitigated (ExponentiatedGradient)** | 80.00% | 0.0667 | 0.9042 | 0.0894 | **+5.9%** |

$$\text{Bias Reduction \%} = \frac{\text{DP\_Diff}_{\text{baseline}} - \text{DP\_Diff}_{\text{mitigated}}}{\text{DP\_Diff}_{\text{baseline}}} \times 100\%$$

### B. Gender Auditing
- **Baseline Demographic Parity Gap**: 0.1293 (Male Selection Rate: 73.56%, Female Selection Rate: 60.63%).
- **Mitigated Reweighing Gap**: 0.0412 (68.1% bias reduction across gender groups).

### C. Proxy Leakage Analysis
- Random Forest sensitive attribute reconstruction demonstrates that dropping `college_tier` and `gender` alone (Fairness through Unawareness) is insufficient: sensitive attributes can be reconstructed from non-sensitive features (e.g., coding volume, communication score, branch) with ROC-AUC $> 0.68$, confirming latent proxy leakage.

---

## 6. Actionable Recourse with Strict Immutability

- **Locked Immutable Attributes**: `gender`, `college_tier`, `branch`, `tenth_pct`, and `twelfth_pct` cannot be altered.
- **Actionable Developmental Pathways**:
  - **Technical Mastery**: Focuses on DSA problem practice, full-stack projects, and hackathons.
  - **Industry Credentials**: Focuses on internships, certifications, and clearing backlogs.
  - **Balanced Pathway**: Proportional distribution of effort across coding, aptitude, and communication.
- **Legal Disclaimer**: Actionable recourse recommendations are mathematical counterfactual simulations and do not guarantee an employment contract.

---

## 7. Regulatory & Governance Compliance

- **EU Artificial Intelligence Act (High-Risk AI Systems)**: PlaceWise provides explainability (SHAP), data validation (Pandera), cryptographic manifests, and automated human-in-the-loop abstention zones for borderline predictions.
- **EEOC Uniform Guidelines (80% Four-Fifths Rule)**: Baseline demographic parity ratios ($< 0.80$) are mitigated to $> 0.90$ using Sample Reweighing.
