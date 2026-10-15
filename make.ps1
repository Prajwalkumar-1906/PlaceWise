<#
.SYNOPSIS
    Cross-platform task runner script for Windows PowerShell.
.DESCRIPTION
    Mirrors all Makefile targets: setup, data, train, audit, app, test, lint, format, clean.
#>

param (
    [Parameter(Position=0)]
    [string]$Target = "help"
)

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$VenvStreamlit = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"
$VenvPytest = Join-Path $PSScriptRoot ".venv\Scripts\pytest.exe"
$VenvRuff = Join-Path $PSScriptRoot ".venv\Scripts\ruff.exe"
$VenvBlack = Join-Path $PSScriptRoot ".venv\Scripts\black.exe"

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
    "app" {
        Write-Host "Launching Streamlit application..." -ForegroundColor Cyan
        & $VenvStreamlit run app/streamlit_app.py
    }
    "test" {
        Write-Host "Running pytest with coverage..." -ForegroundColor Cyan
        & $VenvPytest tests -v --cov=src --cov-report=term-missing
    }
    "lint" {
        Write-Host "Running Ruff and Black lint checks..." -ForegroundColor Cyan
        & $VenvRuff check src tests app
        & $VenvBlack --check src tests app
    }
    "format" {
        Write-Host "Formatting code with Ruff and Black..." -ForegroundColor Cyan
        & $VenvRuff check --fix src tests app
        & $VenvBlack src tests app
    }
    "clean" {
        Write-Host "Cleaning cache files..." -ForegroundColor Cyan
        Get-ChildItem -Path . -Include __pycache__, .pytest_cache, .ruff_cache -Recurse -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
        Write-Host "Clean complete." -ForegroundColor Green
    }
    default {
        Write-Host "Campus Placement Predictor CLI Runner" -ForegroundColor Yellow
        Write-Host "Available targets:"
        Write-Host "  .\make.ps1 setup   - Setup venv and install dependencies"
        Write-Host "  .\make.ps1 data    - Generate synthetic data (unbiased & biased)"
        Write-Host "  .\make.ps1 train   - Train and tune ML models"
        Write-Host "  .\make.ps1 audit   - Run Fairlearn audit & mitigation"
        Write-Host "  .\make.ps1 app     - Start Streamlit UI"
        Write-Host "  .\make.ps1 test    - Run test suite with coverage"
        Write-Host "  .\make.ps1 lint    - Run linters"
        Write-Host "  .\make.ps1 format  - Auto-format code"
        Write-Host "  .\make.ps1 clean   - Remove cache directories"
    }
}
