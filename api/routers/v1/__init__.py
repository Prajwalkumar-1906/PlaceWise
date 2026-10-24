"""API v1 routers."""

from api.routers.v1.explain import router as explain_router
from api.routers.v1.model_info import router as model_info_router
from api.routers.v1.predict import router as predict_router

__all__ = ["explain_router", "model_info_router", "predict_router"]
