"""Configuration loading utility for YAML files."""

from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        yaml.YAMLError: If parsing fails.
    """
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Configuration file not found: {path.resolve()}")
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    try:
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.debug(f"Loaded configuration from {path}")
        return config or {}
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file {path}: {exc}")
        raise
