"""Download Kaggle Campus Recruitment dataset (Ben Roshan) with graceful fallback and clear instructions."""

import argparse
import os
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

KAGGLE_DATASET = "benroshan/factors-affecting-campus-placement"
TARGET_FILENAME = "Placement_Data_Full_Class.csv"


def download_dataset(target_dir: str | Path = "data/raw", optional: bool = False) -> bool:
    """Attempts to download the Kaggle dataset, with fallback instructions if credentials are absent.

    Args:
        target_dir: Directory where raw CSV should reside.
        optional: If True, missing credentials will log info instead of warning.

    Returns:
        bool: True if dataset file exists or was downloaded; False otherwise.
    """
    raw_dir = Path(target_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_file = raw_dir / TARGET_FILENAME

    if target_file.exists():
        logger.info(
            f"Target dataset already exists at: {target_file.resolve()} (Size: {target_file.stat().st_size} bytes)"
        )
        return True

    logger.info(f"Attempting to download '{KAGGLE_DATASET}' via Kaggle API...")

    kaggle_json_user = Path.home() / ".kaggle" / "kaggle.json"
    has_kaggle_env = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    has_kaggle_file = kaggle_json_user.exists()

    if not (has_kaggle_env or has_kaggle_file):
        msg = (
            "\n" + "=" * 80 + "\n"
            "KAGGLE CREDENTIALS NOT DETECTED\n"
            "The real baseline dataset ('Placement_Data_Full_Class.csv' by Ben Roshan)\n"
            "was NOT automatically downloaded because Kaggle credentials were not found.\n\n"
            "TO OBTAIN THE REAL DATASET:\n"
            "Option A (Manual Download - Recommended):\n"
            "  1. Visit: https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement\n"
            "  2. Download 'Placement_Data_Full_Class.csv'\n"
            f"  3. Place it into: {raw_dir.resolve()}/{TARGET_FILENAME}\n\n"
            "Option B (Kaggle API Automation):\n"
            "  1. Generate an API token from your Kaggle Account Settings ('Create New Token').\n"
            "  2. Place 'kaggle.json' in ~/.kaggle/ (or set KAGGLE_USERNAME & KAGGLE_KEY env vars).\n"
            "  3. Re-run: python -m src.data.download\n"
            "=" * 80 + "\n"
        )
        if optional:
            logger.info(msg)
        else:
            logger.warning(msg)
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        logger.info(
            f"Authenticated with Kaggle API. Downloading '{KAGGLE_DATASET}' to {raw_dir}..."
        )
        api.dataset_download_files(KAGGLE_DATASET, path=str(raw_dir), unzip=True)

        if target_file.exists():
            logger.info(
                f"Successfully downloaded and extracted {target_file.name} ({target_file.stat().st_size} bytes)."
            )
            return True
        else:
            logger.warning(
                f"Download completed, but expected file {TARGET_FILENAME} was not found in {raw_dir}."
            )
            return False
    except Exception as e:
        logger.warning(
            f"Failed to download via Kaggle API: {e}\n"
            f"Please manually download 'Placement_Data_Full_Class.csv' from:\n"
            f"https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement\n"
            f"and place it at: {target_file.resolve()}"
        )
        return False


def main() -> None:
    """CLI entrypoint for dataset download."""
    parser = argparse.ArgumentParser(description="Download real Kaggle Campus Recruitment dataset.")
    parser.add_argument(
        "--target-dir", type=str, default="data/raw", help="Directory to save raw dataset."
    )
    parser.add_argument(
        "--optional",
        action="store_true",
        help="Treat missing Kaggle credentials gracefully without warning.",
    )
    args = parser.parse_args()

    download_dataset(target_dir=args.target_dir, optional=args.optional)


if __name__ == "__main__":
    main()
