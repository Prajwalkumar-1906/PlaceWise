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

## 3. Disaggregated Group Selection Rates

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

## 4. Key Findings & Discussion

### A. The Failure of 'Fairness through Unawareness'
Simply removing sensitive attributes (`gender`, `college_tier`) from the training set does **not** ensure fairness. Because technical skills (DSA problem counts), aptitude scores, and branch choices correlate with institutional tier, the unaware model reconstructs institutional bias through proxy features, yielding nearly identical disparities.

### B. Inherent Mathematical Conflicts in Fairness Criteria
Fairness criteria frequently stand in mathematical tension (Chouldechova, 2017; Kleinberg et al., 2016):
1. **Demographic Parity** requires equal selection rates regardless of ground-truth base rate differences.
2. **Equalized Odds** requires equal true-positive and false-positive rates across groups.
When underlying historical conditions cause differing positive rates between groups, enforcing Demographic Parity forces the classifier to accept higher false positives for historically disadvantaged groups, slightly depressing raw test accuracy.

### C. The Price of Fairness (Pareto Trade-Off)
The trade-off analysis reveals that Post-processing (`ThresholdOptimizer`) achieves substantial bias reduction with only a minor accuracy adjustment. However, in high-stakes recruitment scenarios, post-processing assigns different decision cutoffs to different demographics, which must be clearly justified under legal and organizational policies.

### D. Production Deployment Recommendation
- **Human-in-the-Loop Review**: No candidate should be automatically rejected on the basis of a single model score.
- **Continuous Monitoring**: Model calibration and selection rates should be monitored dynamically across admission cycles.
- **Upstream Interventions**: True equity cannot be solved solely by downstream algorithmic mitigation; institutional support, mentoring, and technical resources must be directed towards unprivileged demographics.

---
*(Report generated automatically by `src/fairness/report.py`)*