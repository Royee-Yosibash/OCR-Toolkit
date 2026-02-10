import numpy as np
import pytest

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.config import OCRConfig
from ocr_backbone.ocr import OCRAbstact


class DummyOCR(OCRAbstact):
    """Returns a single BB covering the full sub-image."""

    def __init__(self) -> None:
        self.initialized = True

    def _run_single(self, image: np.ndarray, config: OCRConfig) -> list[BoundingBox]:
        h, w = image.shape[:2]
        return [
            BoundingBox(
                coordinates=((0, 0), (w, h)),
                text="dummy",
                confidence=1.0,
            )
        ]


def test_ocr_cannot_be_instantiated():
    with pytest.raises(TypeError):
        OCRAbstact()


def test_single_cell_grid():
    ocr = DummyOCR()
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    config = OCRConfig(model_name="dummy", grid_rows=1, grid_cols=1)
    result = ocr.get_text_bb(image, config)
    assert len(result) == 1
    assert result[0].coordinates == ((0, 0), (200, 100))


def test_grid_splits_and_remaps():
    ocr = DummyOCR()
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    config = OCRConfig(model_name="dummy", grid_rows=2, grid_cols=2)
    result = ocr.get_text_bb(image, config)
    assert len(result) == 4
    assert result[0].coordinates == ((0, 0), (100, 50))
    assert result[1].coordinates == ((100, 0), (200, 50))
    assert result[2].coordinates == ((0, 50), (100, 100))
    assert result[3].coordinates == ((100, 50), (200, 100))


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
