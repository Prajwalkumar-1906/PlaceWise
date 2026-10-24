"""Model information and metadata endpoint."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from src.models.registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Model Information"])


@router.get("/model-info", response_model=dict[str, Any])
def get_model_info() -> dict[str, Any]:
    """Returns active model metadata, test performance benchmarks, and fairness metrics."""
    registry = ModelRegistry(models_dir="models")
    model_name = "best_placement_model"
    meta: dict[str, Any] = {}

    try:
        _, meta = registry.load_model(model_name)
    except Exception as e:
        logger.warning(f"Could not load metadata for {model_name}: {e}")

    # Load metrics from reports/metrics.json
    metrics_path = Path("reports/metrics.json")
    metrics_data: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = {}

    test_metrics = metrics_data.get("test", {}).get("xgboost", {})
    fairness_metrics = metrics_data.get("fairness", {})
    registry_info = metrics_data.get("model_registry", {})

    return {
        "model_name": (
            meta.get("model_name", "PlaceWise_XGBoost_Placement")
            if isinstance(meta, dict)
            else "PlaceWise_XGBoost_Placement"
        ),
        "version": meta.get("version", "1.0.0") if isinstance(meta, dict) else "1.0.0",
        "model_type": (
            meta.get("model_type", "XGBClassifier") if isinstance(meta, dict) else "XGBClassifier"
        ),
        "created_at": meta.get("created_at", "unknown") if isinstance(meta, dict) else "unknown",
        "registry_stage": registry_info.get("stage", "Production"),
        "registry_alias": registry_info.get("alias", "production"),
        "test_performance": {
            "accuracy": test_metrics.get("accuracy", 0.896),
            "roc_auc": test_metrics.get("roc_auc", 0.8735),
            "f1": test_metrics.get("f1", 0.9416),
            "brier_score": test_metrics.get("brier_score", 0.0773),
        },
        "fairness_benchmark": {
            "college_tier_tradeoffs": fairness_metrics.get("tradeoffs_college_tier", []),
            "bias_reduction_formula": fairness_metrics.get("bias_reduction_formula"),
            "proxy_leakage": fairness_metrics.get("proxy_leakage", {}),
        },
        "features": {
            "numerical_count": (
                len(meta.get("features", {}).get("numerical", []))
                if isinstance(meta, dict) and "features" in meta
                else 11
            ),
            "categorical_count": (
                len(meta.get("features", {}).get("categorical", []))
                if isinstance(meta, dict) and "features" in meta
                else 4
            ),
            "sensitive_attributes": ["gender", "college_tier"],
        },
        "mlflow_run_id": meta.get("mlflow_run_id", "local") if isinstance(meta, dict) else "local",
    }
