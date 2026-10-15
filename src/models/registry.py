"""Model registry for saving, loading, versioning, and cataloging model pipelines and metadata."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Manages serialization and metadata tracking for trained placement and compensation models."""

    def __init__(self, models_dir: str | Path = "models") -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        pipeline: Pipeline,
        model_name: str,
        metadata: dict[str, Any],
    ) -> tuple[Path, Path]:
        """Saves a fitted scikit-learn Pipeline and associated JSON metadata.

        Args:
            pipeline: Fitted scikit-learn Pipeline containing preprocessor and model.
            model_name: Base filename (e.g. 'xgboost_placement').
            metadata: Metadata dictionary containing metrics, features, hyperparameters.

        Returns:
            tuple (model_artifact_path, metadata_path)
        """
        model_path = self.models_dir / f"{model_name}.joblib"
        meta_path = self.models_dir / f"{model_name}_metadata.json"

        # Enrich metadata
        meta = dict(metadata)
        meta["model_name"] = model_name
        meta["saved_at"] = datetime.now(UTC).isoformat()
        meta["artifact_path"] = str(model_path.resolve())

        # Save pipeline via joblib
        joblib.dump(pipeline, model_path)
        logger.info(
            f"Saved model artifact to {model_path.resolve()} ({model_path.stat().st_size} bytes)"
        )

        # Save metadata JSON
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved model metadata to {meta_path.resolve()}")

        return model_path, meta_path

    def load_model(
        self, model_name: str = "best_placement_model"
    ) -> tuple[Pipeline, dict[str, Any]]:
        """Loads a model pipeline and its metadata.

        Args:
            model_name: Base name of the model without extension.

        Returns:
            tuple (loaded_pipeline, metadata_dict)
        """
        model_path = self.models_dir / f"{model_name}.joblib"
        meta_path = self.models_dir / f"{model_name}_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path.resolve()}")

        pipeline = joblib.load(model_path)
        metadata = {}
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)

        logger.info(f"Loaded model '{model_name}' successfully.")
        return pipeline, metadata
