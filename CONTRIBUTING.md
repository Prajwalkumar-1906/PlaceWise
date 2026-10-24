# Contributing to PlaceWise

Thank you for contributing to PlaceWise: Explainable and Fair Campus Placement Predictor.
This project enforces high engineering standards, fairness audits, strict type checking, and zero-compromise documentation accuracy.

---

## 1. Development Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Prajwalkumar-1906/PlaceWise.git
   cd PlaceWise
   ```

2. **Initialize Python environment (Python 3.10+)**:
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify environment**:
   ```bash
   # On Windows PowerShell
   .\make.ps1 test
   # On Linux/macOS
   make test
   ```

---

## 2. Quality Gates & Pre-Commit Standards

Every Pull Request must pass the following four automated gates:

### Gate 1: Unit & Integration Tests ($\ge 80\%$ Coverage)
```bash
# Windows
.\make.ps1 test
# Linux/macOS
make test
```
- Tests must pass without regression.
- Test coverage must remain above $80\%$ across `src/` and `api/`.

### Gate 2: Static Analysis & Code Linting
```bash
# Windows
.\make.ps1 lint
# Linux/macOS
make lint
```
- We use [Ruff](https://github.com/astral-sh/ruff) for linting. All warnings and errors must be resolved.

### Gate 3: Code Formatting
```bash
# Windows
.\make.ps1 format
# Linux/macOS
make format
```
- Enforced via Black and Ruff with 100-character line lengths.

### Gate 4: Strict Type Checking
```bash
# Windows
.\make.ps1 type-check
# Linux/macOS
make type-check
```
- Checked via `mypy` against `src/` and `api/` with zero typing errors.

---

## 3. Data Honesty & Documentation Policy

- **No Hallucinated Benchmarks**: Every metric, AUC value, accuracy percentage, or demographic disparity reported in `README.md`, `MODEL_CARD.md`, and `DATA_CARD.md` must originate directly from executed code recorded in `reports/metrics.json` or `reports/fairness_audit.json`.
- **Fairness & Immutability**: Any modification to actionable recourse algorithms must preserve immutable features (`gender`, `college_tier`, `branch`, `tenth_pct`, `twelfth_pct`).
- **Zero-PII Privacy**: No personally identifiable student information (names, roll numbers, IP addresses) may be committed or stored in plain text. Always compute deterministic SHA-256 hashes for tracking.

---

## 4. Branching & Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature-name`.
2. Commit changes with clear, descriptive commit messages.
3. If introducing a significant architectural decision, document it as an ADR under `docs/adr/`.
4. Ensure all quality gates pass locally before pushing.
5. Open a Pull Request referencing related issues.
