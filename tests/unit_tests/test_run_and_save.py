import json
import unittest

import numpy as np

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr_abstract import OCRAbstact
from ocr_backbone.ocr_result import OCRResult
from utils.run_and_save import run_and_save


class StubOCR(OCRAbstact):

    def __init__(self, config=None) -> None:
        if config is None:
            from ocr_backbone.ocr_config import OCRConfig
            config = OCRConfig(model_name="StubOCR")
        super().__init__(config)

    def _run_single(self, image, model_params) -> OCRResult:
        h, w = image.shape[:2]
        return OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (w, h)), text="stub", confidence=0.99),
        ])


class TestRunAndSave(unittest.TestCase):

    def test_saves_and_returns_result(self, tmp_path=None):
        if tmp_path is None:
            import tempfile, pathlib
            tmp_path = pathlib.Path(tempfile.mkdtemp())

        image = np.zeros((50, 100, 3), dtype=np.uint8)
        ocr = StubOCR()
        save_path = str(tmp_path / "result.json")

        result = run_and_save(image, ocr, save_path)

        self.assertIsInstance(result, OCRResult)
        self.assertEqual(len(result.bounding_boxes), 1)
        self.assertEqual(result.bounding_boxes[0].text, "stub")

        with open(save_path) as f:
            loaded = OCRResult.from_dict(json.load(f))
        self.assertEqual(loaded, result)
