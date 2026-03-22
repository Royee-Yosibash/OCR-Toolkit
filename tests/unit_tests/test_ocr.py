import unittest

import numpy as np

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.input_image import InputImage
from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_backbone.ocr_result import OCRResult


class DummyOCR(OCRAbstract):
    """Returns a single BB covering the full sub-image."""

    def __init__(self, config=None) -> None:
        if config is None:
            config = OCRConfig(model_name="DummyOCR")
        super().__init__(config)

    def _run_single(self, image: np.ndarray, single_run_model_params: dict) -> OCRResult:
        h, w = image.shape[:2]
        return OCRResult(bounding_boxes=[
            BoundingBox(
                coordinates=((0, 0), (w, h)),
                text="dummy",
                confidence=1.0,
            )
        ])


class TestOCR(unittest.TestCase):
    """Tests for OCR abstract class functionality."""

    def test_single_cell_grid(self):
        ocr = DummyOCR()
        ocr.config = OCRConfig(model_name="dummy", grid_rows=1, grid_cols=1)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ocr.get_text_bb(image)
        self.assertEqual(len(result.bounding_boxes), 1)
        self.assertEqual(result.bounding_boxes[0].coordinates, ((0, 0), (200, 100)))

    def test_grid_splits_and_remaps(self):
        ocr = DummyOCR()
        ocr.config = OCRConfig(model_name="dummy", grid_rows=2, grid_cols=2)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ocr.get_text_bb(image)
        bbs = result.bounding_boxes
        self.assertEqual(len(bbs), 4)
        self.assertEqual(bbs[0].coordinates, ((0, 0), (100, 50)))
        self.assertEqual(bbs[1].coordinates, ((100, 0), (200, 50)))
        self.assertEqual(bbs[2].coordinates, ((0, 50), (100, 100)))
        self.assertEqual(bbs[3].coordinates, ((100, 50), (200, 100)))

    def test_bb_validator_filters(self):
        ocr = DummyOCR()
        ocr.config = OCRConfig(
            model_name="dummy",
            grid_rows=2,
            grid_cols=2,
            bb_validator=lambda bb: bb.coordinates[0][0] == 0,
        )
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ocr.get_text_bb(image)
        self.assertEqual(len(result.bounding_boxes), 2)
        self.assertTrue(all(bb.coordinates[0][0] == 0 for bb in result.bounding_boxes))

    def test_bb_validator_none_keeps_all(self):
        ocr = DummyOCR()
        ocr.config = OCRConfig(model_name="dummy", grid_rows=2, grid_cols=2, bb_validator=None)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ocr.get_text_bb(image)
        self.assertEqual(len(result.bounding_boxes), 4)

    def test_split_image_dimensions(self):
        ocr = DummyOCR()
        image = np.zeros((90, 120, 3), dtype=np.uint8)
        input_image = InputImage(image=image)
        grid = ocr._split_image(input_image, rows=3, cols=3)
        self.assertEqual(len(grid), 9)
        for cell in grid:
            self.assertEqual(cell.image.shape[0], 30)
            self.assertEqual(cell.image.shape[1], 40)

    def test_from_config_returns_registered_class(self):
        config = OCRConfig(model_name="DummyOCR")
        ocr = OCRAbstract.from_config(config)
        self.assertIsInstance(ocr, DummyOCR)

    def test_from_config_unknown_model(self):
        config = OCRConfig(model_name="nonexistent")
        with self.assertRaises(ValueError, msg="Unknown model"):
            OCRAbstract.from_config(config)
