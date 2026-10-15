"""Unit tests for DataLoader and train/val/test splitting."""

import pandas as pd

from src.data.loader import DataLoader


def test_loader_auto_generate(tmp_path):
    """Verify that DataLoader loads synthetic data, and auto-generates if absent."""
    loader = DataLoader()
    # Test loading the existing dataset
    df = loader.load_synthetic_data(biased=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "placed" in df.columns
    assert "salary_band" in df.columns

    # Test auto-generation to a non-existent path
    custom_file = tmp_path / "custom_test_synth.csv"
    loader.paths["synthetic_dir"] = str(tmp_path)
    loader.data_cfg["synthetic_file"] = "custom_test_synth.csv"
    df_gen = loader.load_synthetic_data(
        biased=False, auto_generate=True, n_samples=300, random_seed=42
    )
    assert len(df_gen) == 300
    assert custom_file.exists()


def test_train_val_test_split_ratios_and_no_leakage():
    """Verify train/val/test splitting produces disjoint splits with correct ratios and stratification."""
    loader = DataLoader()
    df = loader.load_synthetic_data(
        biased=False, auto_generate=True, n_samples=1000, random_seed=42
    )

    train_df, val_df, test_df = loader.get_train_val_test_split(
        df, target_col="placed", stratify=True, random_seed=42
    )

    total_len = len(df)
    assert len(train_df) + len(val_df) + len(test_df) == total_len

    # Check ratios approximately match 70 / 15 / 15
    assert abs(len(train_df) / total_len - 0.70) < 0.02
    assert abs(len(val_df) / total_len - 0.15) < 0.02
    assert abs(len(test_df) / total_len - 0.15) < 0.02

    # Verify stratification: placement rates should be close
    orig_rate = df["placed"].mean()
    train_rate = train_df["placed"].mean()
    val_rate = val_df["placed"].mean()
    test_rate = test_df["placed"].mean()

    assert abs(train_rate - orig_rate) < 0.03
    assert abs(val_rate - orig_rate) < 0.04
    assert abs(test_rate - orig_rate) < 0.04
