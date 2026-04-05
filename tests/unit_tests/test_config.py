import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ocr_backbone.image_preprocessing import binarize
from ocr_backbone.input_image import InputImage
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


class TestResolvePPMethod(unittest.TestCase):
    """Tests for OCRConfig._resolve_pp_method resolution logic."""

    def test_plain_name_resolves_from_image_preprocessing(self):
        method = OCRConfig._resolve_pp_method({"name": "binarize"})
        self.assertIs(method, binarize)

    def test_plain_name_with_kwargs_returns_partial(self):
        method = OCRConfig._resolve_pp_method(
            {"name": "binarize", "kwargs": {"method": "otsu"}}
        )
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        result = method(InputImage(image=image))
        self.assertEqual(len(result.image.shape), 2)

    def test_dotted_path_resolves_function(self):
        method = OCRConfig._resolve_pp_method(
            {"name": "ocr_backbone.image_preprocessing.binarize"}
        )
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        result = method(InputImage(image=image))
        self.assertEqual(len(result.image.shape), 2)

    def test_dotted_path_with_kwargs(self):
        method = OCRConfig._resolve_pp_method(
            {"name": "ocr_backbone.image_preprocessing.binarize",
             "kwargs": {"method": "otsu"}}
        )
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = method(InputImage(image=image))
        unique = set(np.unique(result.image))
        self.assertTrue(unique.issubset({0, 255}))

    def test_dotted_path_bad_module_raises(self):
        with self.assertRaises(ModuleNotFoundError):
            OCRConfig._resolve_pp_method(
                {"name": "nonexistent_package.some_func"}
            )

    def test_dotted_path_bad_attribute_raises(self):
        with self.assertRaises(AttributeError):
            OCRConfig._resolve_pp_method(
                {"name": "ocr_backbone.image_preprocessing.no_such_func"}
            )

    def test_plain_name_bad_attribute_raises(self):
        with self.assertRaises(AttributeError):
            OCRConfig._resolve_pp_method({"name": "no_such_func"})


class TestFromDictPreprocessMethods(unittest.TestCase):
    """Tests for preprocess_methods resolution in OCRConfig.from_dict."""

    def test_from_dict_resolves_plain_name(self):
        config = OCRConfig.from_dict({
            "model_name": "test",
            "preprocess_methods": [{"name": "binarize"}],
        })
        self.assertEqual(len(config.preprocess_methods), 1)
        self.assertTrue(callable(config.preprocess_methods[0]))

    def test_from_dict_resolves_dotted_path(self):
        config = OCRConfig.from_dict({
            "model_name": "test",
            "preprocess_methods": [
                {"name": "ocr_backbone.image_preprocessing.binarize"}
            ],
        })
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        result = config.preprocess_methods[0](InputImage(image=image))
        self.assertEqual(len(result.image.shape), 2)

    def test_from_dict_passes_through_callables(self):
        config = OCRConfig.from_dict({
            "model_name": "test",
            "preprocess_methods": [binarize],
        })
        self.assertIs(config.preprocess_methods[0], binarize)

    def test_to_dict_round_trips_preprocess_methods(self):
        config = OCRConfig.from_dict({
            "model_name": "test",
            "preprocess_methods": [
                {"name": "binarize", "kwargs": {"method": "otsu"}},
            ],
        })
        data = config.to_dict()
        self.assertEqual(len(data["preprocess_methods"]), 1)
        entry = data["preprocess_methods"][0]
        self.assertIn("name", entry)
        self.assertEqual(entry["kwargs"], {"method": "otsu"})

    def test_update_resolves_dict_preprocess_methods(self):
        config = OCRConfig(model_name="test")
        config.update({
            "preprocess_methods": [{"name": "binarize"}],
        })
        self.assertEqual(len(config.preprocess_methods), 1)
        self.assertTrue(callable(config.preprocess_methods[0]))
