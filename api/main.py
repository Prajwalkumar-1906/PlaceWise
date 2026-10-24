"""Main FastAPI application entrypoint for PlaceWise."""

import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers.health import router as health_router
from api.routers.v1.explain import router as explain_router
from api.routers.v1.model_info import router as model_info_router
from api.routers.v1.predict import router as predict_router
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="PlaceWise Prediction & Explainability API",
    version="0.1.0",
    description=(
        "Production-grade microservice providing explainable and fair campus placement predictions, "
        "uncertainty quantification (HITL abstention zone), and actionable developmental recourse."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for Streamlit frontend and local integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Any) -> Any:
    """Middleware recording request execution latency."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Custom structured response for Pydantic schema validation failures."""
    logger.warning(f"Schema validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "One or more fields in the applicant payload failed schema validation.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Standardized HTTP exception response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTP Error", "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Sanitized internal server error response."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred during processing.",
        },
    )


# Attach routers
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(model_info_router)


@app.get("/", tags=["Root"])
def root() -> dict[str, str]:
    """Root entrypoint linking to interactive documentation."""
    return {
        "service": "PlaceWise Placement Prediction API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
