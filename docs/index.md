# PlaceWise: Explainable and Fair Campus Placement Predictor

Welcome to the technical documentation for **PlaceWise**, a production-grade, fair, and explainable machine learning system designed for university career services and placement decision support.

---

## Key Pillars

1. **Production-Grade Machine Learning**:
   - Stratified, leak-free preprocessing pipelines.
   - 5x5 Repeated Cross-Validation and 1,000-sample bootstrap confidence intervals.
   - SQLite-backed MLflow experiment tracking and automated model registry promotion.

2. **Rigorous Fairness & Auditing**:
   - Demographic Parity and Equalized Odds disparity evaluations with 95% bootstrap intervals.
   - Sensitive attribute proxy leakage detection.
   - Multi-strategy mitigation (Fairness through Unawareness, Sample Reweighing, ThresholdOptimizer, ExponentiatedGradient) and empirical Pareto trade-off curves.
   - Actionable recourse recommendations strictly locking immutable demographic attributes.

3. **Uncertainty & Decision Governance**:
   - Platt & Isotonic probability calibration evaluated via Expected Calibration Error (ECE).
   - Split Conformal Prediction with $\ge 90\%$ marginal coverage guarantee.
   - Automated abstention zone routing ambiguous candidates ($P \in [0.35, 0.65]$) to Human-in-the-Loop career counselors.

4. **Production Architecture**:
   - High-performance FastAPI REST API with Pydantic validation.
   - Zero-PII SHA-256 applicant input hashing and JSONL audit logging.
   - Multi-container Docker Compose deployment (REST API + Streamlit Web Portal).
   - Evidently 0.7 automated drift detection with statistical triggers for automated retraining.

---

## Quick Navigation

- [Deployment Guide](deployment.md): Instructions for running via Docker Compose or local microservices.
- [Architecture Decision Records (ADRs)](adr/0001-mlflow-model-registry.md): Documented architectural trade-offs.
- [Changelog](CHANGELOG.md): Detailed release notes.
- [Contributing](CONTRIBUTING.md): Pre-commit quality gates and contribution standards.
