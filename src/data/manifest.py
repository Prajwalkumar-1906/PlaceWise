"""Data artifact provenance tracking, SHA-256 integrity hashing, and manifest generation."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_file_sha256(file_path: Path | str) -> str:
    """Calculates SHA-256 cryptographic checksum of a file.

    Args:
        file_path: Path to target file.

    Returns:
        Hexadecimal hash string.
    """
    p = Path(file_path)
    if not p.exists():
        return "FILE_NOT_FOUND"

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_data_manifest(
    data_dir: Path | str = "data",
    output_path: Path | str = "reports/data_manifest.json",
) -> dict[str, Any]:
    """Inspects all data files and produces a comprehensive provenance manifest.

    Args:
        data_dir: Root directory containing dataset files.
        output_path: JSON manifest destination path.

    Returns:
        dict: Manifest metadata.
    """
    base_dir = Path(data_dir)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    target_files = [
        base_dir / "synthetic" / "synthetic_placement_data.csv",
        base_dir / "synthetic" / "synthetic_placement_data_biased.csv",
        base_dir / "raw" / "Placement_Data_Full_Class.csv",
    ]

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "hash_algorithm": "SHA-256",
        "datasets": {},
    }

    for fpath in target_files:
        if not fpath.exists():
            continue

        file_hash = compute_file_sha256(fpath)
        file_size = fpath.stat().st_size
        modified_at = datetime.fromtimestamp(fpath.stat().st_mtime, tz=UTC).isoformat()

        row_count = 0
        col_count = 0
        columns = []
        try:
            df = pd.read_csv(fpath)
            row_count = len(df)
            col_count = len(df.columns)
            columns = list(df.columns)
        except Exception:
            pass

        manifest["datasets"][fpath.name] = {
            "relative_path": str(fpath.as_posix()),
            "sha256": file_hash,
            "size_bytes": file_size,
            "row_count": row_count,
            "column_count": col_count,
            "columns": columns,
            "last_modified": modified_at,
        }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"Generated data manifest at {out_file.resolve()} ({len(manifest['datasets'])} datasets profiled)."
    )
    return manifest


if __name__ == "__main__":
    generate_data_manifest()
