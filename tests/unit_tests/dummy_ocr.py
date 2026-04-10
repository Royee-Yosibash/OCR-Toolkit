import numpy as np

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_result import OCRResult


class DummyOCR(OCRAbstract):
    """Returns a single BB covering the full sub-image."""

    def __init__(self, config=None) -> None:
        if config is None:
            config = OCRConfig(model_name="DummyOCR")
        super().__init__(config)

    def _run_single(self, image: np.ndarray, single_run_model_params: dict) -> OCRResult:
        h, w = image.shape[:2]
        return OCRResult(
            bounding_boxes=[
                BoundingBox(
                    coordinates=((0, 0), (w, h)),
                    text="dummy",
                    confidence=1.0,
                )
            ]
        )
