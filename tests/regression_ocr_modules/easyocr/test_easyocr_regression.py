import json
from pathlib import Path

import numpy as np
from PIL import Image

from ocr_backbone.ocr_result import OCRResult
from ocr_modules.easyocr_module import EasyOCRConfig, EasyOCRModule

TESTS_DIR = Path(__file__).parents[2]
EXPECTED_DIR = Path(__file__).parent / "expected"


def test_easyocr_regression_test_image():
    image = np.array(Image.open(TESTS_DIR / "test_image.png"))
    config = EasyOCRConfig(model_name="easyocr", languages=["en"])
    ocr = EasyOCRModule(config=config)

    result = ocr.get_text_bb(image)
    with open(EXPECTED_DIR / "test_image.json") as f:
        expected = OCRResult.from_dict(json.load(f))

    assert result.is_close(expected)
