# ADR 0004: FastAPI Microservice Architecture, Audit Logging with Input SHA-256 Hashing, and Containerized Multi-Service Deployment

## Status
Accepted

## Date
2026-10-24

## Context
PlaceWise was originally built solely as an interactive Streamlit monolithic web application. While effective for localized exploratory analysis, a monolithic Streamlit app does not support:
1. Low-latency, stateless headless inference consumption by external university portals and ERP systems.
2. High-throughput batch inference APIs with structured Pydantic schema validation.
3. Formal audit logging for zero-PII compliance and prediction provenance.
4. Independent scaling of web UI versus inference workloads.

## Decision
We engineered a decoupled microservice architecture:
1. **FastAPI Microservice (`api/`)**:
   - Structured routers: `/health`, `/v1/predict`, `/v1/batch-predict`, `/v1/explain`, `/v1/recourse`, `/v1/models`.
   - Comprehensive request/response Pydantic models with field-level validation and Swagger/OpenAPI interactive documentation (`/docs`, `/redoc`).
   - Lifespan management for singleton artifact loading (placement model, salary model, calibrator, conformal predictor, encoders).
2. **Deterministic Audit Logging (`api/audit.py`)**:
   - Appends structured JSON Lines (`api_audit.jsonl`) for every prediction request.
   - Computes deterministic SHA-256 applicant input feature hash (truncated to 16 hex chars) to guarantee full provenance without storing raw student personal identifiers.
   - Logs timestamp, endpoint, latency in milliseconds, model version, and output probabilities.
3. **Containerized Deployment**:
   - Created optimized multi-stage `Dockerfile` with non-root user execution (`appuser:1000`).
   - Created `docker-compose.yml` orchestrating:
     - `placewise-api` on port `8000`.
     - `placewise-web` (Streamlit) on port `8501`.
     - Dedicated healthchecks, volume mounts for models, logs, and reports, and isolated internal bridge network.

## Consequences

### Positive
- External university information systems can consume predictions via RESTful JSON APIs.
- Full provenance tracing for audits without violating student privacy.
- Decoupled front-end UI and backend inference engine allows independent scalability.

### Trade-offs & Mitigations
- Maintaining two services requires synchronized schemas between Streamlit UI forms and FastAPI Pydantic models.
