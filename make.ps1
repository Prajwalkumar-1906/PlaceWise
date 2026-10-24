<#
.SYNOPSIS
    Cross-platform task runner script for Windows PowerShell.
.DESCRIPTION
    Mirrors all Makefile targets: setup, data, train, audit, evaluate, app, api, monitor, docker-up, test, lint, format, type-check, clean.
#>

param (
    [Parameter(Position=0)]
    [string]$Target = "help"
)

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$VenvStreamlit = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"
$VenvUvicorn = Join-Path $PSScriptRoot ".venv\Scripts\uvicorn.exe"
$VenvPytest = Join-Path $PSScriptRoot ".venv\Scripts\pytest.exe"
$VenvRuff = Join-Path $PSScriptRoot ".venv\Scripts\ruff.exe"
$VenvBlack = Join-Path $PSScriptRoot ".venv\Scripts\black.exe"
$VenvMypy = Join-Path $PSScriptRoot ".venv\Scripts\mypy.exe"
$VenvMlflow = Join-Path $PSScriptRoot ".venv\Scripts\mlflow.exe"

switch ($Target.ToLower()) {
    "setup" {
        Write-Host "Setting up Python virtual environment..." -ForegroundColor Cyan
        python -m venv .venv
        & $VenvPython -m pip install --upgrade pip
        & $VenvPython -m pip install -r requirements.txt
    }
    "data" {
        Write-Host "Running data pipeline (download + synthetic generation)..." -ForegroundColor Cyan
        & $VenvPython -m src.data.download --optional
        & $VenvPython -m src.data.synthetic_generator --samples 5000 --output data/synthetic/synthetic_placement_data.csv
        & $VenvPython -m src.data.synthetic_generator --samples 5000 --inject-bias --output data/synthetic/synthetic_placement_data_biased.csv
    }
    "train" {
        Write-Host "Training models with Optuna hyperparameter optimization..." -ForegroundColor Cyan
        & $VenvPython -m src.models.train --config config/config.yaml
    }
    "audit" {
        Write-Host "Running fairness audit and mitigation analysis..." -ForegroundColor Cyan
        & $VenvPython -m src.fairness.audit --config config/config.yaml
    }
    "evaluate" {
        Write-Host "Running rigorous model evaluation suite (Repeated CV, Bootstrap CIs, Ablation, Robustness)..." -ForegroundColor Cyan
        & $VenvPython -m src.models.evaluation_suite --config config/config.yaml
    }
    "app" {
        Write-Host "Launching Streamlit application..." -ForegroundColor Cyan
        & $VenvStreamlit run app/streamlit_app.py
    }
    "api" {
        Write-Host "Launching FastAPI service on http://localhost:8000..." -ForegroundColor Cyan
        & $VenvUvicorn api.main:app --reload --port 8000
    }
    "monitor" {
        Write-Host "Running Evidently drift monitoring report..." -ForegroundColor Cyan
        & $VenvPython -m src.monitoring.drift --config config/config.yaml
    }
    "docker-up" {
        Write-Host "Starting Docker multi-container stack (API + Streamlit)..." -ForegroundColor Cyan
        docker compose up --build
    }
    "mlflow-ui" {
        Write-Host "Launching MLflow UI on http://localhost:5000..." -ForegroundColor Cyan
        & $VenvMlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
    }
    "test" {
        Write-Host "Running pytest with coverage..." -ForegroundColor Cyan
        & $VenvPytest tests -v --cov=src --cov=api --cov-report=term-missing
    }
    "lint" {
        Write-Host "Running Ruff and Black lint checks..." -ForegroundColor Cyan
        & $VenvRuff check src tests app api
        & $VenvBlack --check src tests app api
    }
    "format" {
        Write-Host "Formatting code with Ruff and Black..." -ForegroundColor Cyan
        & $VenvRuff check --fix src tests app api
        & $VenvBlack src tests app api
    }
    "type-check" {
        Write-Host "Running Mypy static type analysis..." -ForegroundColor Cyan
        & $VenvMypy src api
    }
    "clean" {
        Write-Host "Cleaning cache files..." -ForegroundColor Cyan
        Get-ChildItem -Path . -Include __pycache__, .pytest_cache, .ruff_cache, .mypy_cache -Recurse -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
        Write-Host "Clean complete." -ForegroundColor Green
    }
    default {
        Write-Host "Campus Placement Predictor CLI Runner" -ForegroundColor Yellow
        Write-Host "Available targets:"
        Write-Host "  .\make.ps1 setup      - Setup venv and install dependencies"
        Write-Host "  .\make.ps1 data       - Generate synthetic data (unbiased & biased)"
        Write-Host "  .\make.ps1 train      - Train and tune ML models"
        Write-Host "  .\make.ps1 audit      - Run Fairlearn audit & mitigation"
        Write-Host "  .\make.ps1 evaluate   - Run rigorous evaluation suite (5x5 CV, bootstrap, ablations)"
        Write-Host "  .\make.ps1 app        - Start Streamlit UI"
        Write-Host "  .\make.ps1 api        - Start FastAPI microservice (port 8000)"
        Write-Host "  .\make.ps1 monitor    - Run Evidently drift monitoring"
        Write-Host "  .\make.ps1 docker-up  - Start Docker containers"
        Write-Host "  .\make.ps1 mlflow-ui  - Start local MLflow tracking UI"
        Write-Host "  .\make.ps1 test       - Run test suite with coverage"
        Write-Host "  .\make.ps1 lint       - Run linters"
        Write-Host "  .\make.ps1 format     - Auto-format code"
        Write-Host "  .\make.ps1 type-check - Run static type checking"
        Write-Host "  .\make.ps1 clean      - Remove cache directories"
    }
}
