"""Convenience script for downloading the Kaggle Campus Recruitment dataset."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import download_dataset

if __name__ == "__main__":
    raw_dir = PROJECT_ROOT / "data" / "raw"
    success = download_dataset(target_dir=raw_dir, optional=False)
    if success:
        print(f"Dataset ready at {raw_dir / 'Placement_Data_Full_Class.csv'}")
    else:
        print("Real dataset download skipped or failed; see instructions above.")
