import logging

import numpy as np

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_result import OCRResult

logger = logging.getLogger(__name__)

try:
    import easyocr

    _HAS_EASYOCR = True
except ImportError:
    _HAS_EASYOCR = False
    logger.info("easyocr not installed -- EasyOCRModule will not be available.")


if _HAS_EASYOCR:

    class EasyOCRModule(OCRAbstract):
        """OCR module using the EasyOCR engine."""

        def __init__(self, config: OCRConfig | dict, alias: str = "") -> None:
            """Initialize the EasyOCR reader.

            Args:
                config: An OCRConfig instance or a dict that will be
                    unpacked into one.
                alias: Optional display name used as the label in evaluations.
            """
            super().__init__(config, alias=alias)
            self._reader = easyocr.Reader(self.config.model_params.pop("languages", ["en"]))

        def _run_single(self, image: np.ndarray, single_run_model_params: dict) -> OCRResult:
            """Run EasyOCR on a single image.

            Args:
                image: Input image as a numpy array (H x W x C).
                single_run_model_params: Additional keyword arguments passed to
                    ``easyocr.Reader.readtext``.

            Returns:
                An OCRResult containing detected text regions.
            """
            results = self._reader.readtext(image, **single_run_model_params)
            bboxes = []
            for corners, text, score in results:
                xs = [int(pt[0]) for pt in corners]
                ys = [int(pt[1]) for pt in corners]
                bb = BoundingBox(
                    coordinates=((min(xs), min(ys)), (max(xs), max(ys))),
                    text=text,
                    confidence=score,
                )
                bboxes.append(bb)
            return OCRResult(detections=bboxes)
