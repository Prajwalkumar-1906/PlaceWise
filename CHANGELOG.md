# Changelog

All notable changes to the PlaceWise project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-10-24

### Added
- **MLflow Tracking & Model Registry**: Integrated local SQLite backend (`sqlite:///mlruns/mlflow.db`), logging hyperparameters, metrics, git commit hashes, data hashes, and automated staging/production promotion.
- **Rigorous Evaluation Suite**: 5x5 Repeated Stratified K-Fold Cross-Validation, 1,000-sample bootstrap 95% confidence intervals, paired t-tests, Wilcoxon signed-rank tests, ablation studies across feature categories, and perturbation stress testing (Gaussian noise, missingness).
- **Advanced Fairness & Bias Auditing**:
  - Sensitive attribute proxy leakage detection (`src/fairness/proxy.py`).
  - Vectorized 1,000-sample bootstrap confidence intervals for Demographic Parity and Equalized Odds differences and ratios.
  - Multi-attribute counterfactual fairness audits testing single-attribute perturbations on gender and college tier.
  - Multi-strategy Fairlearn mitigation comparison (Unawareness, Reweighing, ThresholdOptimizer, ExponentiatedGradient) with empirical Pareto trade-off curve generation.
- **Uncertainty Quantification & Human-in-the-Loop (HITL)**:
  - Expected Calibration Error (ECE) metric and Platt / Isotonic probability calibrators using `FrozenEstimator`.
  - Inductive Split Conformal Prediction guaranteeing $\ge 90\%$ marginal coverage with discrete prediction sets ($\{1\}$, $\{0\}$, $\{0, 1\}$, $\emptyset$).
  - Automated abstention zone routing ($[0.35, 0.65]$ calibrated probability) flagging ambiguous applicants for human career counselor review.
- **Actionable Recourse**: Recourse engine generating developmental pathways (Technical, Industry Experience, Balanced) while strictly locking immutable features (`gender`, `college_tier`, `branch`, `tenth_pct`, `twelfth_pct`).
- **FastAPI Microservice Suite**:
  - High-throughput asynchronous REST API (`api/`) with endpoints: `/health`, `/v1/predict`, `/v1/batch-predict`, `/v1/explain`, `/v1/recourse`, `/v1/models`.
  - Zero-PII deterministic SHA-256 applicant input hashing and JSONL audit logging (`logs/api_audit.jsonl`).
  - Multi-container Docker deployment (`Dockerfile`, `docker-compose.yml`) orchestrating API on port 8000 and Streamlit on port 8501.
- **Production Drift Monitoring**: Evidently 0.7 tabular drift analysis, 2-sample Kolmogorov-Smirnov test, Chi-squared test, automated retraining trigger evaluator, and interactive Streamlit monitoring dashboard (`app/pages/5_📈_Monitoring.py`).
- **Streamlit Enhancements**:
  - Batch CSV applicant prediction tab with aggregate metrics and downloadable annotated CSVs.
  - ReportLab PDF student dossier report generator (`src/utils/pdf_generator.py`).
  - Universal provenance footer (`render_provenance_footer()`) displaying model version, training data hash, and calibration status across all pages.
- **Data Engineering Quality**:
  - Pandera DataFrameSchema data validation rules (`src/data/schemas.py`) and automated data quality Markdown report generator.
  - Cryptographic data manifest (`reports/data_manifest.json`) recording row counts, columns, and SHA-256 hashes.
  - Strict conditional training of salary regression models exclusively on placed candidates ($Y=1$).
- **Architecture & Developer Experience**:
  - Architecture Decision Records (`docs/adr/0001` through `0004`).
  - MkDocs documentation site configuration (`mkdocs.yml`).
  - PowerShell automation script (`make.ps1`) mirroring Makefile targets with added `mlflow-ui`, `evaluate`, `api`, `monitor`, `docker-up`, and `type-check`.
  - Full end-to-end integration test suite (`tests/test_integration.py`).

### Changed
- Standardized plotting backend to `matplotlib.use("Agg")` across all modules for headless CI/CD execution.
- Vectorized metric frame calculations in Fairlearn bootstrap confidence interval computations for sub-second execution.
- Updated `pyproject.toml` with strict `mypy`, `ruff`, and `black` configurations, achieving 100% type-checking compliance.

---

## [0.1.0] - Baseline Release
- Initial baseline PlaceWise repository with synthetic student generator, basic XGBoost training, SHAP tree explainability, initial Fairlearn audit, Streamlit web app, and basic unit tests.
