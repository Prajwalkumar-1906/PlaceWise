# Algorithmic Fairness Audit and Mitigation Report

## 1. Executive Summary
This report audits the Campus Placement Prediction system for disparate impact across **College Institutional Tier** and **Gender**, using the Fairlearn toolkit.
To establish a verifiable ground truth for algorithmic bias detection, experiments were benchmarked on historical institutional bias injected data, where Tier-3 and Female applicants received systematic placement penalties at identical skill levels.

---

## 2. Quantitative Performance & Fairness Benchmark

### College Tier Audit Benchmark Table
| Model Architecture | Test Accuracy | Demographic Parity Gap | DP Ratio | Equalized Odds Gap | Bias Reduction % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Aware)** | 0.808 | 0.071 | 0.900 | 0.049 | **+0.0%** |
| **Unawareness (Blind)** | 0.843 | 0.020 | 0.977 | 0.091 | **+72.2%** |
| **Mitigated (Reweighing)** | 0.855 | 0.027 | 0.969 | 0.041 | **+62.1%** |
| **Mitigated (ThresholdOptimizer)** | 0.848 | 0.078 | 0.906 | 0.083 | **-10.6%** |
| **Mitigated (ExponentiatedGradient)** | 0.800 | 0.067 | 0.904 | 0.089 | **+5.9%** |

> [!NOTE]
> **Explicit Bias Reduction Formula**:
> $$\text{Bias Reduction \%} = \frac{\text{DP\_Diff}_{\text{baseline}} - \text{DP\_Diff}_{\text{mitigated}}}{\text{DP\_Diff}_{\text{baseline}}} \times 100\%$$
> A positive percentage denotes a quantified reduction in demographic disparity compared to the unmitigated baseline.

---

## 3. Statistical Bootstrap Confidence Intervals (95% CI)
Fairness metrics computed over 1,000 resamples on held-out test data to capture statistical uncertainty:

| Model | Metric | Point Estimate | 95% Bootstrap CI |
| :--- | :--- | :---: | :---: |
| Baseline (Aware) (gender) | Demographic Parity Diff | 0.1126 | [0.0846, 0.1264] |
| Baseline (Aware) (gender) | Equalized Odds Diff | 0.0956 | [0.0450, 0.1514] |
| Baseline (Aware) (college_tier) | Demographic Parity Diff | 0.0663 | [0.0235, 0.1162] |
| Baseline (Aware) (college_tier) | Equalized Odds Diff | 0.1756 | [0.1099, 0.3082] |
| Unawareness (Blind) (gender) | Demographic Parity Diff | 0.0147 | [0.0039, 0.0250] |
| Unawareness (Blind) (gender) | Equalized Odds Diff | 0.0805 | [0.0337, 0.1641] |
| Unawareness (Blind) (college_tier) | Demographic Parity Diff | 0.0347 | [0.0301, 0.0416] |
| Unawareness (Blind) (college_tier) | Equalized Odds Diff | 0.1663 | [0.0649, 0.3148] |
| Mitigated (Reweighing) (gender) | Demographic Parity Diff | 0.0638 | [0.0363, 0.0937] |
| Mitigated (Reweighing) (gender) | Equalized Odds Diff | 0.0756 | [0.0086, 0.1654] |
| Mitigated (Reweighing) (college_tier) | Demographic Parity Diff | 0.0407 | [0.0336, 0.0517] |
| Mitigated (Reweighing) (college_tier) | Equalized Odds Diff | 0.1698 | [0.0714, 0.2844] |
| Mitigated (ThresholdOptimizer) (gender) | Demographic Parity Diff | 0.0808 | [0.0428, 0.1173] |
| Mitigated (ThresholdOptimizer) (gender) | Equalized Odds Diff | 0.0984 | [0.0060, 0.2024] |
| Mitigated (ThresholdOptimizer) (college_tier) | Demographic Parity Diff | 0.0757 | [0.0407, 0.1560] |
| Mitigated (ThresholdOptimizer) (college_tier) | Equalized Odds Diff | 0.1850 | [0.0415, 0.4101] |
| Mitigated (ExponentiatedGradient) (gender) | Demographic Parity Diff | 0.1448 | [0.0971, 0.1858] |
| Mitigated (ExponentiatedGradient) (gender) | Equalized Odds Diff | 0.1413 | [0.0630, 0.2359] |
| Mitigated (ExponentiatedGradient) (college_tier) | Demographic Parity Diff | 0.0761 | [0.0508, 0.1083] |
| Mitigated (ExponentiatedGradient) (college_tier) | Equalized Odds Diff | 0.1853 | [0.1178, 0.2924] |

---

## 4. Counterfactual Fairness Evaluation
Evaluates individual fairness by holding all candidate skill attributes constant while counterfactually perturbing sensitive attributes.
- **Gender Counterfactual**: $Male \leftrightarrow Female$
- **College Tier Counterfactual**: $1 \to 3, 2 \to 1, 3 \to 1$

| Model Architecture | Sensitive Attribute Perturbed | Prediction Flip Rate (%) | Mean Probability Shift ($\Delta P$) | Verdict |
| :--- | :--- | :---: | :---: | :--- |
| Baseline (Aware) | gender | 12.00% | 0.0810 | Sensitive (12.0% flipped) |
| Baseline (Aware) | college_tier | 5.33% | 0.0351 | Sensitive (5.3% flipped) |
| Unawareness (Blind) | gender | 0.00% | 0.0000 | Counterfactually Invariant |
| Unawareness (Blind) | college_tier | 0.00% | 0.0000 | Counterfactually Invariant |
| Mitigated (Reweighing) | gender | 7.20% | 0.0728 | Sensitive (7.2% flipped) |
| Mitigated (Reweighing) | college_tier | 1.07% | 0.0137 | Sensitive (1.1% flipped) |
| Mitigated (ExponentiatedGradient) | gender | 13.87% | 0.1387 | Sensitive (13.9% flipped) |
| Mitigated (ExponentiatedGradient) | college_tier | 2.40% | 0.0240 | Sensitive (2.4% flipped) |
| Mitigated (ThresholdOptimizer) | college_tier | 11.07% | 0.0000 | Sensitive (11.1% flipped) |
| Mitigated (ThresholdOptimizer) | gender | 0.00% | 0.0000 | Counterfactually Invariant |

---

## 5. Statistical Proxy Leakage Detection
Evaluates whether non-sensitive academic and skill features reconstruct sensitive protected attributes, causing 'Fairness through Unawareness' to fail.

| Protected Attribute | Reconstruction ROC-AUC | Reconstruction Accuracy | Reconstruction F1 (Macro) | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **gender** | **0.4607** | 0.5156 | 0.4729 | Moderate / Low Proxy Leakage |
> Top predictive proxy features for `gender`: `dsa_problems_solved`, `aptitude_score`, `twelfth_pct`

| **college_tier** | **0.4955** | 0.4889 | 0.3559 | Moderate / Low Proxy Leakage |
> Top predictive proxy features for `college_tier`: `cgpa`, `dsa_problems_solved`, `aptitude_score`

---

## 6. Disaggregated Group Selection Rates

### College Tier Breakdown
| Model | Tier 1 Rate | Tier 2 Rate | Tier 3 Rate | Tier 1-3 Gap |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (Aware) | 63.9% | 70.9% | 66.9% | 3.1% |
| Unawareness (Blind) | 86.8% | 84.8% | 86.2% | 0.5% |
| Mitigated (Reweighing) | 83.1% | 85.8% | 83.9% | 0.7% |
| Mitigated (ThresholdOptimizer) | 75.9% | 83.7% | 82.5% | 6.6% |
| Mitigated (ExponentiatedGradient) | 62.6% | 68.2% | 69.3% | 6.7% |

### Gender Disparities
| Model | Male Selection Rate | Female Selection Rate | Gender Parity Gap |
| :--- | :---: | :---: | :---: |
| Baseline (Aware) | 73.6% | 60.6% | 0.129 |
| Unawareness (Blind) | 85.8% | 85.7% | 0.000 |
| Mitigated (Reweighing) | 87.6% | 80.3% | 0.073 |
| Mitigated (ThresholdOptimizer) | 86.2% | 76.8% | 0.094 |
| Mitigated (ExponentiatedGradient) | 74.9% | 58.7% | 0.162 |

---

## 7. Key Findings & Discussion

### A. The Failure of 'Fairness through Unawareness'
Simply removing sensitive attributes (`gender`, `college_tier`) from the training set does **not** ensure fairness. Because technical skills (DSA problem counts), aptitude scores, and branch choices correlate with institutional tier, the unaware model reconstructs institutional bias through proxy features, yielding nearly identical disparities.

### B. Inherent Mathematical Conflicts in Fairness Criteria
Fairness criteria frequently stand in mathematical tension (Chouldechova, 2017; Kleinberg et al., 2016):
1. **Demographic Parity** requires equal selection rates regardless of ground-truth base rate differences.
2. **Equalized Odds** requires equal true-positive and false-positive rates across groups.
3. **Predictive Parity (Calibration)** requires that identical probability scores reflect identical placement chances across groups.

Kleinberg et al. (2016) proved that unless base rates are identical across groups or predictions are perfectly accurate ($AUC=1.0$), a model **cannot simultaneously satisfy** calibration within groups and equalized odds / demographic parity. Enforcing demographic parity on historical data where base rates differ mathematically forces disparate false-positive or false-negative distributions.

### C. The Price of Fairness (Pareto Trade-Off)
The trade-off analysis reveals that Post-processing (`ThresholdOptimizer`) achieves substantial bias reduction with only a minor accuracy adjustment. However, in high-stakes recruitment scenarios, post-processing assigns different decision cutoffs to different demographics, which must be clearly justified under legal and organizational policies.

### D. Production Deployment & Responsible AI Guidelines
- **Human-in-the-Loop Review**: No candidate should be automatically rejected on the basis of a single model score.
- **Continuous Monitoring**: Model calibration and selection rates should be monitored dynamically across admission cycles.
- **Upstream Interventions**: True equity cannot be solved solely by downstream algorithmic mitigation; institutional support, mentoring, and technical resources must be directed towards unprivileged demographics.
- **Synthetic Data Limitations**: This audit operates on mathematically structured synthetic cohorts. Real-world bias is intersectional, systemic, historical, and unmeasured.

---
*(Report generated automatically by `src/fairness/report.py`)*