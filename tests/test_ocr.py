import numpy as np
import pytest

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr import OCRAbstact


class DummyOCR(OCRAbstact):
    def __init__(self) -> None:
        self.initialized = True

    def get_text_bb(self, image: np.ndarray, config: dict) -> list[BoundingBox]:
        return [
            BoundingBox(
                coordinates=((0, 0), (1, 1)),
                text="dummy",
                confidence=1.0,
            )
        ]


def test_ocr_cannot_be_instantiated():
    with pytest.raises(TypeError):
        OCRAbstact()


def test_dummy_ocr():
    ocr = DummyOCR()
    assert ocr.initialized
    result = ocr.get_text_bb(np.zeros((10, 10, 3), dtype=np.uint8), config=dict())
    assert len(result) == 1
    assert result[0].text == "dummy"
