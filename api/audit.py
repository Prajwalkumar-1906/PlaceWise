"""Structured audit logging for prediction provenance and zero-PII compliance."""

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

AUDIT_LOG_DIR = Path("logs")
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "api_audit.jsonl"


def compute_input_hash(data: dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 hash of applicant input features for provenance.

    Args:
        data: Applicant feature dictionary.

    Returns:
        Hex string representing input hash (first 16 characters for brevity).
    """
    sorted_items = sorted([(k, str(v)) for k, v in data.items()])
    encoded = json.dumps(sorted_items, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def log_prediction_event(
    endpoint: str,
    input_data: dict[str, Any],
    prediction_result: dict[str, Any],
    latency_ms: float,
    request_id: str | None = None,
    client_ip: str | None = None,
) -> dict[str, Any]:
    """Writes an immutable JSON audit log record for an inference request.

    Args:
        endpoint: API endpoint name.
        input_data: Applicant feature inputs (hashed, no raw PII logged).
        prediction_result: Model output dictionary.
        latency_ms: Execution duration in milliseconds.
        request_id: Unique request identifier.
        client_ip: Optional client IP address.

    Returns:
        The written audit record.
    """
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    req_id = request_id or str(uuid.uuid4())
    inp_hash = compute_input_hash(input_data)
    timestamp = datetime.now(UTC).isoformat()

    record = {
        "timestamp": timestamp,
        "request_id": req_id,
        "endpoint": endpoint,
        "input_hash": inp_hash,
        "placement_probability": prediction_result.get("placement_probability"),
        "placement_prediction": prediction_result.get("placement_prediction"),
        "confidence_band": prediction_result.get("confidence_band"),
        "needs_human_review": prediction_result.get("needs_human_review", False),
        "salary_band": prediction_result.get("salary_band"),
        "latency_ms": round(latency_ms, 2),
        "client_ip": client_ip or "internal",
    }

    try:
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to append to audit log: {e}")

    return record


def read_recent_audit_logs(limit: int = 50) -> list[dict[str, Any]]:
    """Reads the most recent audit records.

    Args:
        limit: Number of records to return.

    Returns:
        list of audit log records in reverse chronological order.
    """
    if not AUDIT_LOG_FILE.exists():
        return []

    records: list[dict[str, Any]] = []
    try:
        with open(AUDIT_LOG_FILE, encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines[-limit:]):
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to read audit log: {e}")

    return records
