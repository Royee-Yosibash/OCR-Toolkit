import numpy as np
import pytest

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_abstract import OCRAbstact
from ocr_backbone.ocr_result import OCRResult


class DummyOCR(OCRAbstact):
    """Returns a single BB covering the full sub-image."""

    def __init__(self) -> None:
        super().__init__()
        self.initialized = True

    def _run_single(self, image: np.ndarray, model_params: dict) -> OCRResult:
        h, w = image.shape[:2]
        return OCRResult(bounding_boxes=[
            BoundingBox(
                coordinates=((0, 0), (w, h)),
                text="dummy",
                confidence=1.0,
            )
        ])


def test_ocr_cannot_be_instantiated():
    with pytest.raises(TypeError):
        OCRAbstact()


def test_single_cell_grid():
    ocr = DummyOCR()
    ocr.config = OCRConfig(model_name="dummy", grid_rows=1, grid_cols=1)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = ocr.get_text_bb(image)
    assert len(result.bounding_boxes) == 1
    assert result.bounding_boxes[0].coordinates == ((0, 0), (200, 100))


def test_grid_splits_and_remaps():
    ocr = DummyOCR()
    ocr.config = OCRConfig(model_name="dummy", grid_rows=2, grid_cols=2)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = ocr.get_text_bb(image)
    bbs = result.bounding_boxes
    assert len(bbs) == 4
    assert bbs[0].coordinates == ((0, 0), (100, 50))
    assert bbs[1].coordinates == ((100, 0), (200, 50))
    assert bbs[2].coordinates == ((0, 50), (100, 100))
    assert bbs[3].coordinates == ((100, 50), (200, 100))


def test_bb_validator_filters():
    ocr = DummyOCR()
    ocr.config = OCRConfig(
        model_name="dummy",
        grid_rows=2,
        grid_cols=2,
        bb_validator=lambda bb: bb.coordinates[0][0] == 0,
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = ocr.get_text_bb(image)
    assert len(result.bounding_boxes) == 2
    assert all(bb.coordinates[0][0] == 0 for bb in result.bounding_boxes)


def test_bb_validator_none_keeps_all():
    ocr = DummyOCR()
    ocr.config = OCRConfig(model_name="dummy", grid_rows=2, grid_cols=2, bb_validator=None)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = ocr.get_text_bb(image)
    assert len(result.bounding_boxes) == 4


def test_split_image_dimensions():
    ocr = DummyOCR()
    image = np.zeros((90, 120, 3), dtype=np.uint8)
    grid = ocr._split_image(image, rows=3, cols=3)
    assert len(grid) == 3
    assert len(grid[0]) == 3
    for row in grid:
        for cell in row:
            assert cell.shape[0] == 30
            assert cell.shape[1] == 40


def test_from_config_returns_registered_class():
    config = OCRConfig(model_name="DummyOCR")
    ocr = OCRAbstact.from_config(config)
    assert isinstance(ocr, DummyOCR)


def test_from_config_unknown_model():
    config = OCRConfig(model_name="nonexistent")
    with pytest.raises(ValueError, match="Unknown model"):
        OCRAbstact.from_config(config)
