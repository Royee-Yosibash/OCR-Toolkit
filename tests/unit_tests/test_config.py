import json
import tempfile
import unittest
from pathlib import Path

from ocr_backbone.ocr_config import OCRConfig, load_config


class TestOCRConfig(unittest.TestCase):
    """Tests for OCRConfig and load_config."""

    def test_ocr_config_defaults(self):
        config = OCRConfig(model_name="test_model")
        self.assertEqual(config.model_name, "test_model")
        self.assertEqual(config.model_params, {})

    def test_ocr_config_with_params(self):
        config = OCRConfig(model_name="easyocr", model_params={"lang": "en"})
        self.assertEqual(config.model_params["lang"], "en")

    def test_load_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "config.json"
            config_file.write_text(
                json.dumps({"model_name": "easyocr", "model_params": {"threshold": 0.5}})
            )
            config = load_config(config_file)
            self.assertEqual(config.model_name, "easyocr")
            self.assertEqual(config.model_params["threshold"], 0.5)

    def test_load_config_missing_params(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "config.json"
            config_file.write_text(json.dumps({"model_name": "easyocr"}))
            config = load_config(config_file)
            self.assertEqual(config.model_params, {})

    def test_load_config_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/path.json")

    def test_load_config_missing_model_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "config.json"
            config_file.write_text(json.dumps({"model_params": {}}))
            with self.assertRaises(KeyError):
                load_config(config_file)
