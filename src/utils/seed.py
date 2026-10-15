"""Random seed initialization for reproducibility across numpy, random, and OS."""

import os
import random

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Sets deterministic seeds across Python standard library and NumPy.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    logger.info(f"Global random seed set to {seed}")
