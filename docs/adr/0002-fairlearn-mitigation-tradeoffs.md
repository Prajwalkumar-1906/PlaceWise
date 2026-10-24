# ADR 0002: Fairlearn Mitigation Strategies, Multi-Attribute Auditing, and Pareto Trade-Off Analysis

## Status
Accepted

## Date
2026-10-24

## Context
Campus placement predictive systems carry intrinsic risks of algorithmic bias across sensitive protected attributes:
1. **Gender (`gender`)**: Systemic historical imbalances in engineering disciplines and recruitment.
2. **Socio-geographic proxy (`college_tier`)**: Placement disparities reflecting institutional hiring patterns rather than individual student aptitude.

In algorithmic fairness, Kleinberg's Impossibility Theorem mathematically demonstrates that an imperfect predictor cannot simultaneously satisfy Demographic Parity (equal selection rate across groups) and Equalized Odds (equal true positive and false positive rates) unless base rates are identical or accuracy is 100%.
Furthermore, naive bias mitigation techniques can degrade overall classification accuracy or introduce reverse discrimination.

## Decision
We adopted a multi-strategy auditing and mitigation pipeline leveraging Microsoft Fairlearn:
1. **Multi-Strategy Comparison**:
   - **Baseline (Aware)**: Default XGBoost with demographic attributes present.
   - **Fairness through Unawareness (Blind)**: Retraining after strictly dropping protected attributes (`gender`, `college_tier`).
   - **Sample Reweighing (Pre-processing)**: Computing inverse marginal probability weights $w_i = \frac{P(Y=y) \cdot P(A=a)}{P(Y=y, A=a)}$ to balance training data without distorting decision boundaries.
   - **Threshold Optimizer (Post-processing)**: Applying group-specific decision thresholds to equalize opportunity or selection rates.
   - **Exponentiated Gradient (In-processing)**: Formulating fairness-constrained optimization over randomized ensemble reductions subject to equalized odds or demographic parity constraints.
2. **Empirical Pareto Trade-off Analysis**:
   Plotting Accuracy vs. Demographic Parity difference across all candidate strategies. We do not arbitrarily enforce a single technique; instead, we expose the empirical Pareto frontier.
3. **Audit Results on `college_tier`**:
   - Baseline: Accuracy = 0.8080, Demographic Parity Difference = 0.0710 (0.0% bias reduction).
   - Unawareness: Accuracy = 0.8427, DP Difference = 0.0197 (72.2% bias reduction).
   - Reweighing: Accuracy = 0.8547, DP Difference = 0.0269 (62.1% bias reduction).
   - ThresholdOptimizer: Accuracy = 0.8480, DP Difference = 0.0785 (-10.6% bias reduction).
   - ExponentiatedGradient: Accuracy = 0.8000, DP Difference = 0.0668 (5.9% bias reduction).

## Consequences

### Positive
- Transparent disclosure of empirical trade-offs: Reweighing yields superior Pareto efficiency (retaining 85.5% accuracy while cutting DP difference by 62.1%).
- Auditing includes 1,000-sample bootstrap 95% confidence intervals to ensure observed disparity reductions are statistically significant rather than sampling artifacts.
- Integration of proxy leakage detection (random forest sensitive attribute reconstruction) prevents hidden bias from infiltrating unmitigated pipelines.

### Trade-offs & Mitigations
- ExponentiatedGradient yields randomized predictors which require deterministic seed pinning for reproducibility.
- Post-processing (ThresholdOptimizer) requires access to sensitive attributes at test/inference time, which may violate certain institutional privacy policies. Unawareness and Reweighing are preferred when test-time sensitive attribute collection is prohibited.
