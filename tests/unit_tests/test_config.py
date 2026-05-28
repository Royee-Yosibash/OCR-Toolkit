import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from ocr_backbone import image_preprocessing
from ocr_backbone.image_preprocessing import binarize
from ocr_backbone.input_image import InputImage
from ocr_backbone.ocr_config import OCRConfig, load_config
from utils.callable_descriptors import resolve_callable_descriptor
from utils.json_utils import save_json
from utils.serialize_utils import TYPE_KEY, SerializableClass


def resolve_pp(descriptor):
    """Resolve a preprocessing descriptor using the same default module ``OCRConfig`` uses."""
    return resolve_callable_descriptor(descriptor, default_module=image_preprocessing)


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
            config_file_path = Path(tmp_dir) / "config.json"
            save_json(config_file_path, {"model_name": "easyocr", "model_params": {"threshold": 0.5}})
            config = load_config(config_file_path)
            self.assertEqual(config.model_name, "easyocr")
            self.assertEqual(config.model_params["threshold"], 0.5)

    def test_load_config_missing_params(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file_path = Path(tmp_dir) / "config.json"
            save_json(config_file_path, {"model_name": "easyocr"})
            config = load_config(config_file_path)
            self.assertEqual(config.model_params, {})

    def test_load_config_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/path.json")

    def test_load_config_missing_model_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file_path = Path(tmp_dir) / "config.json"
            save_json(config_file_path, {"model_params": {}})
            with self.assertRaises(TypeError):
                load_config(config_file_path)


class TestResolvePPMethod(unittest.TestCase):
    """Tests for preprocessing descriptor resolution via ``_resolve_callable_descriptor``."""

    def test_plain_name_resolves_from_image_preprocessing(self):
        method = resolve_pp({"name": "binarize"})
        self.assertIs(method, binarize)

    def test_plain_name_with_kwargs_returns_partial(self):
        method = resolve_pp({"name": "binarize", "kwargs": {"method": "otsu"}})
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        result = method(InputImage(image=image))
        self.assertEqual(len(result.image.shape), 2)

    def test_dotted_path_resolves_function(self):
        method = resolve_pp({"name": "ocr_backbone.image_preprocessing.binarize"})
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        result = method(InputImage(image=image))
        self.assertEqual(len(result.image.shape), 2)

    def test_dotted_path_with_kwargs(self):
        method = resolve_pp({"name": "ocr_backbone.image_preprocessing.binarize", "kwargs": {"method": "otsu"}})
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = method(InputImage(image=image))
        unique = set(np.unique(result.image))
        self.assertTrue(unique.issubset({0, 255}))

    def test_dotted_path_bad_module_raises(self):
        with self.assertRaises(ModuleNotFoundError):
            resolve_pp({"name": "nonexistent_package.some_func"})

    def test_dotted_path_bad_attribute_raises(self):
        with self.assertRaises(AttributeError):
            resolve_pp({"name": "ocr_backbone.image_preprocessing.no_such_func"})

    def test_plain_name_bad_attribute_raises(self):
        with self.assertRaises(AttributeError):
            resolve_pp({"name": "no_such_func"})


class TestFromDictPreprocessMethods(unittest.TestCase):
    """Tests for preprocess_methods resolution in OCRConfig.from_dict."""

    def test_from_dict_resolves_plain_name(self):
        config = OCRConfig.from_dict(
            {
                "model_name": "test",
                "preprocess_methods": [{"name": "binarize"}],
            }
        )
        self.assertEqual(len(config.preprocess_methods), 1)
        self.assertTrue(callable(config.preprocess_methods[0]))

    def test_from_dict_resolves_dotted_path(self):
        config = OCRConfig.from_dict(
            {
                "model_name": "test",
                "preprocess_methods": [{"name": "ocr_backbone.image_preprocessing.binarize"}],
            }
        )
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        result = config.preprocess_methods[0](InputImage(image=image))
        self.assertEqual(len(result.image.shape), 2)

    def test_from_dict_passes_through_callables(self):
        config = OCRConfig.from_dict(
            {
                "model_name": "test",
                "preprocess_methods": [binarize],
            }
        )
        self.assertIs(config.preprocess_methods[0], binarize)

    def test_to_dict_round_trips_preprocess_methods(self):
        config = OCRConfig.from_dict(
            {
                "model_name": "test",
                "preprocess_methods": [
                    {"name": "binarize", "kwargs": {"method": "otsu"}},
                ],
            }
        )
        data = config.to_dict()
        self.assertEqual(len(data["preprocess_methods"]), 1)
        entry = data["preprocess_methods"][0]
        self.assertIn("name", entry)
        self.assertEqual(entry["kwargs"], {"method": "otsu"})

    def test_update_resolves_dict_preprocess_methods(self):
        config = OCRConfig(model_name="test")
        config.update(
            {
                "preprocess_methods": [{"name": "binarize"}],
            }
        )
        self.assertEqual(len(config.preprocess_methods), 1)
        self.assertTrue(callable(config.preprocess_methods[0]))

    def test_from_dict_resolves_external_module(self):
        """Import a preprocessing function from a temporary Python module."""
        module_src = textwrap.dedent("""\
            import numpy as np
            from ocr_backbone.input_image import InputImage

            def halve_height(input_image: InputImage) -> InputImage:
                img = input_image.image
                h = img.shape[0] // 2
                return InputImage(
                    image=img[:h],
                    x_offset=input_image.x_offset,
                    y_offset=input_image.y_offset,
                )
        """)

        with tempfile.TemporaryDirectory() as tmp_dir:
            module_file = Path(tmp_dir) / "custom_pp.py"
            module_file.write_text(module_src)

            sys.path.insert(0, tmp_dir)
            try:
                config = OCRConfig.from_dict(
                    {
                        "model_name": "test",
                        "preprocess_methods": [
                            {"name": "custom_pp.halve_height"},
                        ],
                    }
                )

                self.assertEqual(len(config.preprocess_methods), 1)
                image = np.zeros((100, 200, 3), dtype=np.uint8)
                result = config.preprocess_methods[0](InputImage(image=image))
                self.assertEqual(result.image.shape, (50, 200, 3))
                self.assertEqual(result.x_offset, 0)
                self.assertEqual(result.y_offset, 0)
            finally:
                sys.path.remove(tmp_dir)
                sys.modules.pop("custom_pp", None)


class TestOCRConfigSerializableIntegration(unittest.TestCase):
    """Verifies OCRConfig's participation in the SerializableClass hierarchy."""

    def test_is_registered_in_serializable_registry(self):
        self.assertIs(SerializableClass._registry.get("OCRConfig"), OCRConfig)

    def test_to_dict_emits_type_discriminator(self):
        config = OCRConfig(model_name="easyocr")
        self.assertEqual(config.to_dict()[TYPE_KEY], "OCRConfig")

    def test_create_roundtrip_via_serializable(self):
        original = OCRConfig(model_name="easyocr", model_params={"lang": "en"})
        restored = SerializableClass.create(original.to_dict())
        self.assertIsInstance(restored, OCRConfig)
        self.assertEqual(restored.model_name, "easyocr")
        self.assertEqual(restored.model_params, {"lang": "en"})

    def test_from_dict_ignores_unknown_keys(self):
        config = OCRConfig.from_dict({"model_name": "easyocr", "alias": "Baseline"})
        self.assertEqual(config.model_name, "easyocr")
        self.assertFalse(hasattr(config, "alias"))
