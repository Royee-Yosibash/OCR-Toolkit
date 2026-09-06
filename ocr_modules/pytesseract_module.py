"""OCR module using the pytesseract engine."""

import importlib.util
import logging

import numpy as np

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_result import OCRResult
from utils.lazy_import import LazyModule

logger = logging.getLogger(__name__)

_HAS_PYTESSERACT = importlib.util.find_spec("pytesseract") is not None
if not _HAS_PYTESSERACT:
    logger.info("pytesseract not installed -- PytesseractModule will not be available.")

pytesseract = LazyModule("pytesseract")


if _HAS_PYTESSERACT:

    class PytesseractModule(OCRAbstract):
        """OCR module using the pytesseract (Tesseract) engine."""

        _INIT_PARAM_KEYS = frozenset({"lang"})

        def __init__(self, config: OCRConfig | dict) -> None:
            """Initialize the pytesseract reader.

            Args:
                config: An OCRConfig instance or a dict that will be
                    unpacked into one. Supported model_params keys:
                    ``lang`` (default ``"eng"``).
            """
            super().__init__(config)
            self._lang = self.config.model_params.get("lang", "eng")

        def _run_single(self, image: np.ndarray, single_run_model_params: dict) -> OCRResult:
            """Run pytesseract on a single image.

            Args:
                image: Input image as a numpy array (H x W x C).
                single_run_model_params: Additional keyword arguments passed to
                    ``pytesseract.image_to_data``.

            Returns:
                An OCRResult containing detected text regions.
            """
            data = pytesseract.image_to_data(
                image,
                lang=self._lang,
                output_type=pytesseract.Output.DICT,
                **single_run_model_params,
            )
            bboxes = []
            for i, text in enumerate(data["text"]):
                if not text or not text.strip():
                    continue
                conf = float(data["conf"][i])
                if conf < 0:
                    continue
                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])
                bb = BoundingBox(
                    coordinates=((max(x, 0), max(y, 0)), (max(x + w, 0), max(y + h, 0))),
                    text=text,
                    confidence=conf / 100.0,
                )
                bboxes.append(bb)
            return OCRResult(detections=bboxes)
