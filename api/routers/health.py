"""Health check endpoint for container orchestrators and load balancers."""

import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter

from src.models.registry import ModelRegistry

router = APIRouter(tags=["Health"])

START_TIME = time.time()


@router.get("/health", response_model=dict[str, Any])
def health_check() -> dict[str, Any]:
    """Returns microservice health status, uptime, and model readiness."""
    uptime = time.time() - START_TIME
    model_loaded = False
    model_name = "None"
    version = "unknown"

    try:
        registry = ModelRegistry(models_dir="models")
        _, meta = registry.load_model("best_placement_model")
        model_loaded = True
        model_name = (
            meta.get("model_name", "PlaceWise_XGBoost_Placement")
            if isinstance(meta, dict)
            else "PlaceWise_XGBoost_Placement"
        )
        version = meta.get("version", "1.0.0") if isinstance(meta, dict) else "1.0.0"
    except Exception:
        model_loaded = False

    return {
        "status": "healthy" if model_loaded else "degraded",
        "service": "PlaceWise Placement Prediction API",
        "version": "0.1.0",
        "model_loaded": model_loaded,
        "model_name": model_name,
        "model_version": version,
        "uptime_seconds": round(uptime, 1),
        "timestamp": datetime.now(UTC).isoformat(),
    }
