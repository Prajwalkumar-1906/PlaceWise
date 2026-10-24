"""Prediction endpoints for single student and batch inference."""

import time

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.audit import compute_input_hash, log_prediction_event
from api.schemas.predict import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    StudentPredictionRequest,
    StudentPredictionResponse,
)
from src.models.predict import PlacementPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Prediction"])

# Singleton predictor instance
_predictor: PlacementPredictor | None = None


def get_predictor() -> PlacementPredictor:
    """Returns singleton PlacementPredictor instance."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = PlacementPredictor(
                placement_model_name="best_placement_model",
                salary_model_name="salary_band_model",
                models_dir="models",
            )
        except Exception as e:
            logger.error(f"Failed to initialize PlacementPredictor: {e}")
            raise HTTPException(
                status_code=500, detail=f"Model service initialization failed: {e}"
            ) from e
    return _predictor


@router.post("/predict", response_model=StudentPredictionResponse)
def predict_single(
    student: StudentPredictionRequest, request: Request
) -> StudentPredictionResponse:
    """Computes placement likelihood, risk rating, salary band, and HITL flags for a student profile."""
    predictor = get_predictor()
    start_time = time.perf_counter()
    student_dict = student.model_dump()

    try:
        result = predictor.predict_student(student_dict)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}") from e

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    inp_hash = compute_input_hash(student_dict)
    client_ip = request.client.host if request.client else "unknown"

    # Log to provenance audit log
    log_prediction_event(
        endpoint="/v1/predict",
        input_data=student_dict,
        prediction_result=result,
        latency_ms=latency_ms,
        client_ip=client_ip,
    )

    model_ver = (
        predictor.placement_meta.get("version", "1.0.0")
        if isinstance(predictor.placement_meta, dict)
        else "1.0.0"
    )

    return StudentPredictionResponse(
        placement_probability=result["placement_probability"],
        placement_prediction=result["placement_prediction"],
        risk_label=result["risk_label"],
        risk_color=result["risk_color"],
        salary_band=result["salary_band"],
        confidence_band=result["confidence_band"],
        needs_human_review=result["needs_human_review"],
        hitl_recommendation=result["hitl_recommendation"],
        hitl_status=result.get("hitl_status", "UNKNOWN"),
        input_hash=inp_hash,
        model_version=model_ver,
        latency_ms=round(latency_ms, 2),
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_endpoint(
    batch: BatchPredictionRequest, request: Request
) -> BatchPredictionResponse:
    """Runs batch inference for up to 500 candidate profiles."""
    predictor = get_predictor()
    start_time = time.perf_counter()

    student_records = [s.model_dump() for s in batch.students]
    df_batch = pd.DataFrame(student_records)

    try:
        df_scored = predictor.predict_batch(df_batch)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}") from e

    batch_latency = (time.perf_counter() - start_time) * 1000.0
    model_ver = (
        predictor.placement_meta.get("version", "1.0.0")
        if isinstance(predictor.placement_meta, dict)
        else "1.0.0"
    )

    responses: list[StudentPredictionResponse] = []
    placed_count = 0
    unplaced_count = 0
    review_count = 0

    for i, row in df_scored.iterrows():
        orig_dict = student_records[int(i)]
        prob = float(row["predicted_placement_probability"])
        pred = int(row["predicted_placement"])
        conf_band = str(row["confidence_band"])
        review_flag = bool(row["needs_human_review"])
        salary = str(row.get("predicted_salary_band", "Not Placed"))

        if pred == 1:
            placed_count += 1
        else:
            unplaced_count += 1

        if review_flag:
            review_count += 1

        # Determine risk
        if prob >= 0.75:
            r_label, r_color = "High Placement Likelihood", "green"
        elif prob >= 0.45:
            r_label, r_color = "Competitive / Moderate Risk", "orange"
        else:
            r_label, r_color = "High Risk / Attention Required", "red"

        item_res = StudentPredictionResponse(
            placement_probability=prob,
            placement_prediction=pred,
            risk_label=r_label,
            risk_color=r_color,
            salary_band=salary,
            confidence_band=conf_band,
            needs_human_review=review_flag,
            hitl_recommendation="Batch evaluated profile",
            hitl_status="BATCH_EVALUATED",
            input_hash=compute_input_hash(orig_dict),
            model_version=model_ver,
            latency_ms=round(batch_latency / len(batch.students), 2),
        )
        responses.append(item_res)

    # Log batch audit event
    log_prediction_event(
        endpoint="/v1/predict/batch",
        input_data={"batch_size": len(batch.students)},
        prediction_result={"placed_count": placed_count, "review_count": review_count},
        latency_ms=batch_latency,
        client_ip=request.client.host if request.client else "unknown",
    )

    return BatchPredictionResponse(
        total_records=len(responses),
        placed_count=placed_count,
        unplaced_count=unplaced_count,
        human_review_required_count=review_count,
        batch_latency_ms=round(batch_latency, 2),
        predictions=responses,
    )
