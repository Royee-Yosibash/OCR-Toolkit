import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_result import OCRResult
from ocr_modules.easyocr_module import EasyOCRModule

TESTS_DIR = Path(__file__).parents[2]
EXPECTED_DIR = Path(__file__).parent / "expected"


class TestEasyOCRRegression(unittest.TestCase):
    """Regression tests for the EasyOCR module."""

    def test_regression_test_image(self):
        image = np.array(Image.open(TESTS_DIR / "test_image.png"))
        config = OCRConfig(model_name="easyocr", model_params={"languages": ["en"]})
        ocr = EasyOCRModule(config=config)
        result = ocr.get_text_bb(image)
        with open(EXPECTED_DIR / "test_image.json") as f:
            expected = OCRResult.from_dict(json.load(f))

        confidence_tolerance = 0.5  # TODO: High tolerance due to CPU/GPU machines giving different outputs. Fix.
        self.assertTrue(
            result.is_close(expected, confidence_tolerance=confidence_tolerance),
            msg=f"Expected: {expected}\n Result: {result}",
        )
