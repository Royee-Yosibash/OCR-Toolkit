import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_result import OCRResult
from ocr_modules.easyocr_module import EasyOCRModule
from tests.consts import TEST_IMAGE_PATH
from utils.json_utils import load_json

EXPECTED_DIR = Path(__file__).parent / "expected"


class TestEasyOCRRegression(unittest.TestCase):
    """Regression tests for the EasyOCR module."""

    def test_regression_test_image(self):
        image = np.array(Image.open(TEST_IMAGE_PATH))
        config = OCRConfig(model_name="easyocr", model_params={"languages": ["en"], "gpu": False})
        ocr = EasyOCRModule(config=config)
        result = ocr.get_text_detections(image)

        expected_result_path = EXPECTED_DIR / "test_image.json"
        expected = OCRResult.from_dict(load_json(expected_result_path))

        confidence_tolerance = 0.5  # TODO: High tolerance due to CPU/GPU machines giving different outputs. Fix.
        self.assertTrue(
            result.is_close(expected, confidence_tolerance=confidence_tolerance),
            msg=f"Expected: {expected}\n Result: {result}",
        )
