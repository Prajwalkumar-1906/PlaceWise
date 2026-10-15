.PHONY: help setup data train audit app test lint format clean

PYTHON ?= python
VENV_DIR ?= .venv

ifeq ($(OS),Windows_NT)
    BIN_DIR = $(VENV_DIR)/Scripts
    VENV_PYTHON = $(BIN_DIR)/python.exe
    VENV_STREAMLIT = $(BIN_DIR)/streamlit.exe
    VENV_PYTEST = $(BIN_DIR)/pytest.exe
    VENV_RUFF = $(BIN_DIR)/ruff.exe
    VENV_BLACK = $(BIN_DIR)/black.exe
else
    BIN_DIR = $(VENV_DIR)/bin
    VENV_PYTHON = $(BIN_DIR)/python
    VENV_STREAMLIT = $(BIN_DIR)/streamlit
    VENV_PYTEST = $(BIN_DIR)/pytest
    VENV_RUFF = $(BIN_DIR)/ruff
    VENV_BLACK = $(BIN_DIR)/black
endif

help:
	@echo "Available make commands:"
	@echo "  make setup   - Create virtual environment and install requirements"
	@echo "  make data    - Download real dataset (if available) and generate synthetic data"
	@echo "  make train   - Train baselines and tuned XGBoost models with full evaluation"
	@echo "  make audit   - Run comprehensive Fairlearn audit and generate fairness reports"
	@echo "  make app     - Run the Streamlit interactive dashboard"
	@echo "  make test    - Run test suite with coverage report"
	@echo "  make lint    - Run ruff and black linting checks"
	@echo "  make format  - Auto-format code with ruff and black"
	@echo "  make clean   - Remove cache, logs, and temporary artifacts"

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

app:
	$(VENV_STREAMLIT) run app/streamlit_app.py

test:
	$(VENV_PYTEST) tests -v --cov=src --cov-report=term-missing

lint:
	$(VENV_RUFF) check src tests app
	$(VENV_BLACK) --check src tests app

format:
	$(VENV_RUFF) check --fix src tests app
	$(VENV_BLACK) src tests app

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
