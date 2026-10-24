# ADR 0001: MLflow Experiment Tracking, Local SQLite Backend, and Model Registry

## Status
Accepted

## Date
2026-10-24

## Context
PlaceWise evaluates candidate machine learning models (Logistic Regression, Random Forest, LightGBM, CatBoost, and tuned XGBoost) across both tabular placement prediction and regression salary estimation. 
Previously, model runs were logged ad-hoc to local console logs and dumped directly to `.pkl` pickle files without systematic versioning, lineage tracking, or artifact checksum verification. 

Key challenges:
1. Difficulty in reproducing exact historical runs due to unrecorded hyperparameters and data splits.
2. Inability to trace model artifacts back to git commit hashes and training data SHA-256 hashes.
3. Lack of a formal promotion workflow distinguishing experimental models from production candidates.

## Decision
We implemented a self-contained, enterprise-grade MLflow tracking architecture:
1. **Local SQLite Backend**: Configured MLflow tracking URI to `sqlite:///mlruns/mlflow.db` and artifact storage in `mlruns/artifacts`. This eliminates remote cloud dependencies while maintaining full relational querying and concurrent access safety.
2. **Deterministic Metadata Logging**: Each run automatically records:
   - Git commit hash via `git rev-parse HEAD`.
   - Training dataset SHA-256 checksum.
   - Complete parameter sets (including hyperparameters, random seeds, and preprocessing steps).
   - Evaluation metrics: accuracy, ROC-AUC, PR-AUC, F1-score, Brier score, and ECE.
3. **Model Registry Promotion**: Integrated programmatic registration under `placewise-placement-predictor` and `placewise-salary-regressor`. Implemented automatic staging transitions and strict validation rules before promoting to `"Production"`.
4. **Cloudpickle Serialization**: Pinned `SERIALIZATION_FORMAT_CLOUDPICKLE` to safely handle custom Scikit-Learn transformers, custom encoders, and calibrated pipelines without unpickle failures.

## Consequences

### Positive
- Fully reproducible runs with immutable data and git hash linkages.
- Built-in MLflow UI (`make mlflow-ui`) accessible at `http://localhost:5000` for visual hyperparameter comparison and metric parallel coordinate charts.
- Clear separation between experimental runs and promoted production artifacts.

### Trade-offs & Mitigations
- Local SQLite database must be excluded from git (`.gitignore`) to avoid bloating version control.
- Requires pinning `mlflow>=2.14.0` in `requirements.txt`.
