# PlaceWise Production Deployment Guide

This guide details how to deploy PlaceWise as a containerized microservice suite or as local production services.

---

## 1. Architecture Overview

```
                      +-----------------------------+
                      |   Reverse Proxy / Ingress   |
                      |   (Nginx / ALB / Traefik)   |
                      +--------------+--------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
      +------------------+                       +-------------------+
      |   FastAPI REST   |                       |    Streamlit UI   |
      |   Inference API  |                       |     Web Portal    |
      |   (Port: 8000)   |                       |    (Port: 8501)   |
      +--------+---------+                       +---------+---------+
               |                                           |
               +---------------------+---------------------+
                                     |
                                     v
                       +---------------------------+
                       |    Persisted Artifacts    |
                       | models/  reports/  logs/  |
                       +---------------------------+
```

---

## 2. Docker Compose Deployment (Recommended)

### Prerequisites
- Docker Engine $\ge 24.0$
- Docker Compose v2

### Quickstart
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Prajwalkumar-1906/PlaceWise.git
   cd PlaceWise
   ```

2. Build and start both services:
   ```bash
   docker compose up --build -d
   ```

3. Verify service health:
   ```bash
   docker compose ps
   ```

4. Service Endpoints:
   - **Streamlit Web Portal**: [http://localhost:8501](http://localhost:8501)
   - **FastAPI OpenAPI Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **FastAPI Healthcheck**: [http://localhost:8000/health](http://localhost:8000/health)

5. Stopping Services:
   ```bash
   docker compose down
   ```

---

## 3. Local Production Execution

### Prerequisites
- Python 3.10+ (tested up to 3.14)
- Virtual environment with pinned dependencies:
  ```bash
  python -m venv .venv
  # Windows
  .\.venv\Scripts\activate
  # Linux/macOS
  source .venv/bin/activate

  pip install -r requirements.txt
  ```

### Step 1: Execute Full Pipeline
Generate data, train production models, evaluate benchmarks, run fairness audit, and generate drift baselines:
```bash
python -m src.data.generate_synthetic
python -m src.data.loader
python -m src.models.train
python -m src.models.evaluate
python -m src.fairness.audit
python -m src.fairness.report
```

### Step 2: Launch FastAPI Inference Microservice
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Step 3: Launch Streamlit Web UI
```bash
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 4. Environment Variables & Configuration

Key configuration parameters can be set in `config/config.yaml` or overridden via environment variables:

| Variable / Key | Default | Description |
| :--- | :--- | :--- |
| `API_PORT` | `8000` | FastAPI listening port |
| `API_HOST` | `0.0.0.0` | FastAPI binding host |
| `STREAMLIT_PORT` | `8501` | Streamlit listening port |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlruns/mlflow.db` | MLflow experiment tracking database |
| `AUDIT_LOG_DIR` | `logs/` | Structured JSONL prediction audit logs |

---

## 5. Security & Privacy Hardening

1. **Zero-PII Audit Logging**: All API inference events log deterministic SHA-256 hashes of input features instead of applicant names or identifiable IDs (`api/audit.py`).
2. **Non-Root Docker Execution**: The Docker container executes under a dedicated unprivileged user `appuser` (UID: 1000).
3. **CORS Configuration**: Default FastAPI CORS allows origin restriction for production environments via `CORSMiddleware`.
