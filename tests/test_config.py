import json

import pytest

from ocr_backbone.config import OCRConfig, load_config


def test_ocr_config_defaults():
    config = OCRConfig(model_name="test_model")
    assert config.model_name == "test_model"
    assert config.model_params == {}


def test_ocr_config_with_params():
    config = OCRConfig(model_name="easyocr", model_params={"lang": "en"})
    assert config.model_params["lang"] == "en"


def test_load_config(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"model_name": "easyocr", "model_params": {"threshold": 0.5}})
    )
    config = load_config(config_file)
    assert config.model_name == "easyocr"
    assert config.model_params["threshold"] == 0.5


def test_load_config_missing_params(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"model_name": "easyocr"}))
    config = load_config(config_file)
    assert config.model_params == {}


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.json")


def test_load_config_missing_model_name(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"model_params": {}}))
    with pytest.raises(KeyError):
        load_config(config_file)
