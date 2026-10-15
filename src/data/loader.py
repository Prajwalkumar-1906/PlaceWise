"""Dataset loader providing unified access to synthetic and real datasets, split generation, and schema checks."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.synthetic_generator import SyntheticPlacementGenerator
from src.data.validation import validate_dataframe
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Handles loading, splitting, and validating datasets for training and evaluation."""

    def __init__(self, config_path: str | Path = "config/config.yaml") -> None:
        """Initializes DataLoader with project configuration.

        Args:
            config_path: Path to config.yaml.
        """
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.paths = self.config.get("paths", {})
        self.data_cfg = self.config.get("data", {})
        self.split_cfg = self.config.get("split", {})

    def load_synthetic_data(
        self,
        biased: bool = False,
        auto_generate: bool = True,
        n_samples: int = 5000,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """Loads the synthetic placement dataset. If not found and auto_generate=True, generates it.

        Args:
            biased: Whether to load the biased synthetic dataset.
            auto_generate: If True, generates the dataset if file is absent.
            n_samples: Number of samples if generating.
            random_seed: Random seed if generating.

        Returns:
            pd.DataFrame: Validated synthetic dataset.
        """
        filename = (
            self.data_cfg.get("synthetic_biased_file", "synthetic_placement_data_biased.csv")
            if biased
            else self.data_cfg.get("synthetic_file", "synthetic_placement_data.csv")
        )
        file_path = Path(self.paths.get("synthetic_dir", "data/synthetic")) / filename

        if not file_path.exists():
            if auto_generate:
                logger.info(f"{file_path} not found. Generating on the fly...")
                generator = SyntheticPlacementGenerator()
                df = generator.generate(
                    n_samples=n_samples, random_seed=random_seed, inject_bias=biased
                )
                file_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(file_path, index=False)
                logger.info(f"Saved generated synthetic dataset to {file_path}")
                return df
            else:
                raise FileNotFoundError(f"Synthetic dataset file not found at {file_path}")

        logger.info(f"Loading synthetic dataset from {file_path}")
        df = pd.read_csv(file_path)

        is_valid, errors = validate_dataframe(df, require_targets=True)
        if not is_valid:
            raise ValueError(f"Loaded synthetic dataset failed schema validation: {errors}")

        return df

    def load_real_data(self) -> pd.DataFrame | None:
        """Loads real Kaggle Campus Recruitment dataset if available.

        Returns:
            pd.DataFrame | None: Raw dataframe if found, None if file does not exist.
        """
        filename = self.data_cfg.get("raw_file", "Placement_Data_Full_Class.csv")
        file_path = Path(self.paths.get("raw_dir", "data/raw")) / filename

        if not file_path.exists():
            logger.info(
                f"Real dataset not found at {file_path}. Run 'make data' or check instructions."
            )
            return None

        logger.info(f"Loading real baseline dataset from {file_path}")
        return pd.read_csv(file_path)

    def get_train_val_test_split(
        self,
        df: pd.DataFrame,
        target_col: str = "placed",
        stratify: bool = True,
        random_seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs stratified train/val/test split to guarantee zero data leakage.

        Args:
            df: Input dataset.
            target_col: Target column name for stratification.
            stratify: Whether to perform stratified split.
            random_seed: Seed for split reproducibility.

        Returns:
            tuple (train_df, val_df, test_df)
        """
        train_ratio = self.split_cfg.get("train_size", 0.70)
        val_ratio = self.split_cfg.get("val_size", 0.15)
        test_ratio = self.split_cfg.get("test_size", 0.15)

        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

        strat_target = df[target_col] if stratify and target_col in df.columns else None

        # First split off train
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            stratify=strat_target,
            random_state=random_seed,
        )

        # Then split remaining into val and test
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        temp_strat = temp_df[target_col] if stratify and target_col in temp_df.columns else None

        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_ratio,
            stratify=temp_strat,
            random_state=random_seed,
        )

        logger.info(
            f"Dataset split completed: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
            f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
            f"Test={len(test_df)} ({len(test_df)/len(df):.1%})"
        )

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )
