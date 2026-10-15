import pytest
import yaml

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


def test_seed_everything():
    """Verify seed_everything executes without error."""
    seed_everything(123)


def test_logger_file_output(tmp_path):
    """Verify get_logger creates a file handler when log_file is specified."""
    log_file = tmp_path / "test.log"
    logger = get_logger("test_custom_logger", log_file=log_file)
    logger.info("Test message to file")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test message to file" in content


def test_load_config_nonexistent():
    """Verify FileNotFoundError is raised when config file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_path/fake_config.yaml")


def test_load_config_invalid_yaml(tmp_path):
    """Verify YAMLError is raised for malformed YAML."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: [unclosed list", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_config(bad_yaml)
