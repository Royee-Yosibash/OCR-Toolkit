import unittest
from functools import partial

import numpy as np

from ocr_backbone.image_preprocessing import binarize, grid_split_image
from ocr_backbone.input_image import InputImage
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_backbone.ocr_config import VALID_PP_RETURN_TYPES, OCRConfig
from ocr_backbone.polygon import Polygon
from tests.unit_tests.dummy_ocr import DummyOCR
from utils.callable_descriptors import validate_callable_signature


def _validate_pp(method):
    """Validate a preprocessing callable using the same arguments OCRConfig.validate uses."""
    validate_callable_signature(
        method,
        expected_first_param=InputImage,
        valid_returns=VALID_PP_RETURN_TYPES,
    )


def grid_m_n(m, n):
    """Build a grid_split_image callable bound to the given grid size.

    Args:
        m: Number of rows.
        n: Number of columns.

    Returns:
        A partial wrapping grid_split_image with the grid kwarg bound.
    """
    return partial(grid_split_image, grid=(m, n))


def keep_left_edge(detection: Polygon) -> bool:
    """Detection validator that keeps only detections whose first coordinate has x == 0.

    Args:
        detection: The polygon to test.

    Returns:
        True if the detection's first coordinate is on the left edge.
    """
    return detection.coordinates[0][0] == 0


class TestOCR(unittest.TestCase):
    """Tests for OCR abstract class functionality."""

    def test_no_preprocessing_single_cell(self):
        ocr = DummyOCR()
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ocr.get_text_detections(image)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.detections[0].coordinates, ((0, 0), (200, 100)))

    def test_grid_splits_and_remaps(self):
        ocr = DummyOCR()
        ocr.config = OCRConfig(model_name="dummy", preprocess_methods=[grid_m_n(2, 2)])
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ocr.get_text_detections(image)
        bbs = result.detections
        self.assertEqual(len(bbs), 4)
        self.assertEqual(bbs[0].coordinates, ((0, 0), (100, 50)))
        self.assertEqual(bbs[1].coordinates, ((100, 0), (200, 50)))
        self.assertEqual(bbs[2].coordinates, ((0, 50), (100, 100)))
        self.assertEqual(bbs[3].coordinates, ((100, 50), (200, 100)))

    def test_detection_validators_filter(self):
        ocr = DummyOCR()
        ocr.config = OCRConfig(
            model_name="dummy",
            preprocess_methods=[grid_m_n(2, 2)],
            detection_validators=[keep_left_edge],
        )
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ocr.get_text_detections(image)
        self.assertEqual(len(result.detections), 2)
        self.assertTrue(all(bb.coordinates[0][0] == 0 for bb in result.detections))

    def test_no_validators_keeps_all(self):
        ocr = DummyOCR()
        ocr.config = OCRConfig(
            model_name="dummy",
            preprocess_methods=[grid_m_n(2, 2)],
        )
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ocr.get_text_detections(image)
        self.assertEqual(len(result.detections), 4)

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
        result = ocr.get_text_detections(image)
        self.assertEqual(len(result.detections), 4)

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
            OCRConfig.from_dict(
                {
                    "model_name": "dummy",
                    "preprocess_methods": [{"name": "nonexistent_func"}],
                }
            )


class _RecordingOCR(DummyOCR):
    """DummyOCR variant that records ``single_run_model_params`` for inspection."""

    _INIT_PARAM_KEYS = frozenset({"init_only"})

    def __init__(self, config=None, alias: str = "") -> None:
        super().__init__(config=config, alias=alias)
        self.received_params: list[dict] = []

    def _run_single(self, image, single_run_model_params):
        self.received_params.append(single_run_model_params)
        return super()._run_single(image, single_run_model_params)


class TestInitParamFiltering(unittest.TestCase):
    """Tests for _INIT_PARAM_KEYS filtering and config-mutation safety."""

    def test_init_only_key_stripped_before_run_single(self):
        config = OCRConfig(
            model_name="DummyOCR",
            model_params={"init_only": "value", "call_kwarg": 42},
        )
        ocr = _RecordingOCR(config=config)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        ocr.get_text_detections(image)
        self.assertEqual(len(ocr.received_params), 1)
        self.assertNotIn("init_only", ocr.received_params[0])
        self.assertEqual(ocr.received_params[0].get("call_kwarg"), 42)

    def test_config_model_params_not_mutated(self):
        original = {"init_only": "value", "call_kwarg": 42}
        config = OCRConfig(model_name="DummyOCR", model_params=dict(original))
        ocr = _RecordingOCR(config=config)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        ocr.get_text_detections(image)
        self.assertEqual(ocr.config.model_params, original)

    def test_to_dict_round_trip_preserves_init_keys(self):
        config = OCRConfig(
            model_name="DummyOCR",
            model_params={"init_only": "value", "call_kwarg": 42},
        )
        _RecordingOCR(config=config)
        restored = OCRConfig.from_dict(config.to_dict())
        self.assertEqual(restored.model_params, {"init_only": "value", "call_kwarg": 42})


class TestValidatePPSignature(unittest.TestCase):
    """Tests for ``_validate_callable_signature`` applied to preprocessing methods."""

    def test_valid_annotated_function(self):
        def good(img: InputImage) -> InputImage:
            return img

        _validate_pp(good)

    def test_valid_list_return(self):
        def good(img: InputImage) -> list[InputImage]:
            return [img]

        _validate_pp(good)

    def test_valid_union_return(self):
        def good(img: InputImage) -> InputImage | list[InputImage]:
            return img

        _validate_pp(good)

    def test_unannotated_first_param_raises(self):
        def bad(img):
            return img

        with self.assertRaises(TypeError):
            _validate_pp(bad)

    def test_missing_return_annotation_raises(self):
        def bad(img: InputImage):
            return img

        with self.assertRaises(TypeError):
            _validate_pp(bad)

    def test_wrong_first_param_annotation_raises(self):
        def bad(img: np.ndarray) -> InputImage:
            return InputImage(image=img)

        with self.assertRaises(TypeError):
            _validate_pp(bad)

    def test_wrong_return_annotation_raises(self):
        def bad(img: InputImage) -> np.ndarray:
            return img.image

        with self.assertRaises(TypeError):
            _validate_pp(bad)

    def test_no_params_raises(self):
        def bad() -> InputImage:
            return InputImage(image=np.zeros((1, 1)))

        with self.assertRaises(TypeError):
            _validate_pp(bad)

    def test_partial_with_valid_remaining_param(self):
        _validate_pp(partial(binarize, method="otsu"))

    def test_rejects_at_construction_time(self):
        def bad(img: np.ndarray) -> InputImage:
            return InputImage(image=img)

        with self.assertRaises(TypeError):
            OCRConfig(model_name="dummy", preprocess_methods=[bad])
