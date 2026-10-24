.PHONY: help setup data train audit evaluate app api monitor docker-up test lint format type-check clean mlflow-ui

PYTHON ?= python
VENV_DIR ?= .venv

ifeq ($(OS),Windows_NT)
    BIN_DIR = $(VENV_DIR)/Scripts
    VENV_PYTHON = $(BIN_DIR)/python.exe
    VENV_STREAMLIT = $(BIN_DIR)/streamlit.exe
    VENV_UVICORN = $(BIN_DIR)/uvicorn.exe
    VENV_PYTEST = $(BIN_DIR)/pytest.exe
    VENV_RUFF = $(BIN_DIR)/ruff.exe
    VENV_BLACK = $(BIN_DIR)/black.exe
    VENV_MYPY = $(BIN_DIR)/mypy.exe
    VENV_MLFLOW = $(BIN_DIR)/mlflow.exe
else
    BIN_DIR = $(VENV_DIR)/bin
    VENV_PYTHON = $(BIN_DIR)/python
    VENV_STREAMLIT = $(BIN_DIR)/streamlit
    VENV_UVICORN = $(BIN_DIR)/uvicorn
    VENV_PYTEST = $(BIN_DIR)/pytest
    VENV_RUFF = $(BIN_DIR)/ruff
    VENV_BLACK = $(BIN_DIR)/black
    VENV_MYPY = $(BIN_DIR)/mypy
    VENV_MLFLOW = $(BIN_DIR)/mlflow
endif

help:
	@echo "Available make commands:"
	@echo "  make setup      - Create virtual environment and install requirements"
	@echo "  make data       - Download real dataset (if available) and generate synthetic data"
	@echo "  make train      - Train baselines and tuned XGBoost models with full evaluation"
	@echo "  make audit      - Run comprehensive Fairlearn audit and generate fairness reports"
	@echo "  make evaluate   - Run rigorous evaluation suite (5x5 CV, bootstrap, ablations, stress)"
	@echo "  make app        - Run the Streamlit interactive dashboard"
	@echo "  make api        - Run FastAPI microservice on http://localhost:8000"
	@echo "  make monitor    - Run Evidently drift monitoring report"
	@echo "  make docker-up  - Build and launch multi-container stack (API + Streamlit)"
	@echo "  make test       - Run test suite with coverage report"
	@echo "  make lint       - Run ruff and black linting checks"
	@echo "  make format     - Auto-format code with ruff and black"
	@echo "  make type-check - Run mypy static type checking"
	@echo "  make mlflow-ui  - Launch MLflow tracking UI on port 5000"
	@echo "  make clean      - Remove cache, logs, and temporary artifacts"

setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

data:
	$(VENV_PYTHON) -m src.data.download --optional
	$(VENV_PYTHON) -m src.data.synthetic_generator --samples 5000 --output data/synthetic/synthetic_placement_data.csv
	$(VENV_PYTHON) -m src.data.synthetic_generator --samples 5000 --inject-bias --output data/synthetic/synthetic_placement_data_biased.csv

train:
	$(VENV_PYTHON) -m src.models.train --config config/config.yaml

audit:
	$(VENV_PYTHON) -m src.fairness.audit --config config/config.yaml

evaluate:
	$(VENV_PYTHON) -m src.models.evaluation_suite --config config/config.yaml

app:
	$(VENV_STREAMLIT) run app/streamlit_app.py

api:
	$(VENV_UVICORN) api.main:app --reload --port 8000

monitor:
	$(VENV_PYTHON) -m src.monitoring.drift --config config/config.yaml

docker-up:
	docker compose up --build

mlflow-ui:
	$(VENV_MLFLOW) ui --backend-store-uri sqlite:///mlruns.db --port 5000

test:
	$(VENV_PYTEST) tests -v --cov=src --cov=api --cov-report=term-missing

lint:
	$(VENV_RUFF) check src tests app api
	$(VENV_BLACK) --check src tests app api

format:
	$(VENV_RUFF) check --fix src tests app api
	$(VENV_BLACK) src tests app api

type-check:
	$(VENV_MYPY) src api

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
