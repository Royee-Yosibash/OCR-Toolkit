import unittest
from functools import partial

import numpy as np

from ocr_backbone.image_preprocessing import binarize, grid_split_image
from ocr_backbone.input_image import InputImage
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_backbone.ocr_config import OCRConfig
from tests.unit_tests.dummy_ocr import DummyOCR


def grid_m_n(m, n):
    """Build a grid_split_image callable bound to the given grid size.

    Args:
        m: Number of rows.
        n: Number of columns.

    Returns:
        A partial wrapping grid_split_image with the grid kwarg bound.
    """
    return partial(grid_split_image, grid=(m, n))


class TestOCR(unittest.TestCase):
    """Tests for OCR abstract class functionality."""

    def test_no_preprocessing_single_cell(self):
        ocr = DummyOCR()
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ocr.get_text_bb(image)
        self.assertEqual(len(result.bounding_boxes), 1)
        self.assertEqual(result.bounding_boxes[0].coordinates, ((0, 0), (200, 100)))

    def test_grid_splits_and_remaps(self):
        ocr = DummyOCR()
        ocr.config = OCRConfig(model_name="dummy", preprocess_methods=[grid_m_n(2, 2)])
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
            preprocess_methods=[grid_m_n(2, 2)],
            bb_validator=lambda bb: bb.coordinates[0][0] == 0,
        )
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ocr.get_text_bb(image)
        self.assertEqual(len(result.bounding_boxes), 2)
        self.assertTrue(all(bb.coordinates[0][0] == 0 for bb in result.bounding_boxes))

    def test_bb_validator_none_keeps_all(self):
        ocr = DummyOCR()
        ocr.config = OCRConfig(
            model_name="dummy",
            preprocess_methods=[grid_m_n(2, 2)],
            bb_validator=None,
        )
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ocr.get_text_bb(image)
        self.assertEqual(len(result.bounding_boxes), 4)

    def test_preprocess_binarize_then_grid(self):
        ocr = DummyOCR()
        ocr.config = OCRConfig(
            model_name="dummy",
            preprocess_methods=[
                partial(binarize, method="otsu"),
                grid_m_n(2, 2),
            ],
        )
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        result = ocr.get_text_bb(image)
        self.assertEqual(len(result.bounding_boxes), 4)

    def test_from_config_returns_registered_class(self):
        config = OCRConfig(model_name="DummyOCR")
        ocr = OCRAbstract.from_config(config)
        self.assertIsInstance(ocr, DummyOCR)

    def test_from_config_unknown_model(self):
        config = OCRConfig(model_name="nonexistent")
        with self.assertRaises(ValueError, msg="Unknown model"):
            OCRAbstract.from_config(config)

    def test_unknown_preprocess_method_raises(self):
        with self.assertRaises(AttributeError):
            OCRConfig.from_dict({
                "model_name": "dummy",
                "preprocess_methods": [{"name": "nonexistent_func"}],
            })


class TestValidatePPSignature(unittest.TestCase):
    """Tests for OCRConfig._validate_pp_signature."""

    def test_valid_annotated_function(self):
        def good(img: InputImage) -> InputImage:
            return img
        OCRConfig._validate_pp_signature(good)

    def test_valid_list_return(self):
        def good(img: InputImage) -> list[InputImage]:
            return [img]
        OCRConfig._validate_pp_signature(good)

    def test_valid_union_return(self):
        def good(img: InputImage) -> InputImage | list[InputImage]:
            return img
        OCRConfig._validate_pp_signature(good)

    def test_unannotated_first_param_raises(self):
        def bad(img):
            return img
        with self.assertRaises(TypeError):
            OCRConfig._validate_pp_signature(bad)

    def test_missing_return_annotation_raises(self):
        def bad(img: InputImage):
            return img
        with self.assertRaises(TypeError):
            OCRConfig._validate_pp_signature(bad)

    def test_wrong_first_param_annotation_raises(self):
        def bad(img: np.ndarray) -> InputImage:
            return InputImage(image=img)
        with self.assertRaises(TypeError):
            OCRConfig._validate_pp_signature(bad)

    def test_wrong_return_annotation_raises(self):
        def bad(img: InputImage) -> np.ndarray:
            return img.image
        with self.assertRaises(TypeError):
            OCRConfig._validate_pp_signature(bad)

    def test_no_params_raises(self):
        def bad() -> InputImage:
            return InputImage(image=np.zeros((1, 1)))
        with self.assertRaises(TypeError):
            OCRConfig._validate_pp_signature(bad)

    def test_partial_with_valid_remaining_param(self):
        OCRConfig._validate_pp_signature(partial(binarize, method="otsu"))

    def test_rejects_at_construction_time(self):
        def bad(img: np.ndarray) -> InputImage:
            return InputImage(image=img)

        with self.assertRaises(TypeError):
            OCRConfig(model_name="dummy", preprocess_methods=[bad])
