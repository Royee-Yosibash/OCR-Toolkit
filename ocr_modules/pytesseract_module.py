"""OCR module using the pytesseract engine."""

import logging

import numpy as np

from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_result import OCRResult

logger = logging.getLogger(__name__)

try:
    import pytesseract

    _HAS_PYTESSERACT = True
except ImportError:
    _HAS_PYTESSERACT = False
    logger.info("pytesseract not installed -- PytesseractModule will not be available.")


if _HAS_PYTESSERACT:

    class PytesseractModule(OCRAbstract):
        """OCR module using the pytesseract (Tesseract) engine.

        Skeleton implementation. The underlying engine is not yet wired up;
        the class exists so it can be registered as an ``OCRAbstract``
        subclass and selected via configuration.
        """

        def __init__(self, config: OCRConfig | dict, alias: str = "") -> None:
            """Initialize the pytesseract module.

            Args:
                config: An OCRConfig instance or a dict that will be
                    unpacked into one.
                alias: Optional display name used as the label in evaluations.
            """
            super().__init__(config, alias=alias)
            self._reader = pytesseract

        def _run_single(self, image: np.ndarray, single_run_model_params: dict) -> OCRResult:
            """Run pytesseract on a single image.

            Not yet implemented.

            Args:
                image: Input image as a numpy array (H x W x C).
                single_run_model_params: Additional keyword arguments for the
                    pytesseract call.

            Returns:
                An OCRResult containing detected text regions.

            Raises:
                NotImplementedError: Always; the engine integration is pending.
            """
            raise NotImplementedError("PytesseractModule._run_single is not yet implemented.")
