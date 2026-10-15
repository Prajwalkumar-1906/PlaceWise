"""Unit tests for the Kaggle dataset download helper."""

from unittest.mock import patch

from src.data.download import download_dataset


def test_download_dataset_already_exists(tmp_path):
    """Verify that download_dataset returns True immediately if file already exists."""
    target_file = tmp_path / "Placement_Data_Full_Class.csv"
    target_file.write_text("dummy,data\n1,2\n")

    result = download_dataset(target_dir=tmp_path, optional=True)
    assert result is True


def test_download_dataset_missing_credentials(tmp_path):
    """Verify that download_dataset gracefully returns False when credentials are missing."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        result = download_dataset(target_dir=tmp_path, optional=True)
        assert result is False


def test_download_dataset_kaggle_exception(tmp_path):
    """Verify that download_dataset handles API exceptions gracefully."""
    with patch.dict("os.environ", {"KAGGLE_USERNAME": "test_user", "KAGGLE_KEY": "test_key"}):
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_api:
            instance = mock_api.return_value
            instance.authenticate.side_effect = Exception("Auth failed")
            result = download_dataset(target_dir=tmp_path, optional=True)
            assert result is False
