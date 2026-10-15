"""Data ingestion, synthetic generation, validation, and loading."""

from src.data.loader import DataLoader
from src.data.synthetic_generator import SyntheticPlacementGenerator
from src.data.validation import StudentRecord, validate_dataframe

__all__ = [
    "DataLoader",
    "StudentRecord",
    "SyntheticPlacementGenerator",
    "validate_dataframe",
]
