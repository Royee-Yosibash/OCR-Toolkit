from dataclasses import dataclass, field
from typing import List, Union
import easyocr
import numpy as np

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr_abstract import OCRAbstact
from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_result import OCRResult


@dataclass
class EasyOCRConfig(OCRConfig):
    """Configuration specific to the EasyOCR engine.

    Args:
        languages: List of language codes for the EasyOCR reader.
    """

    languages: List[str] = field(default_factory=lambda: ["en"])


class EasyOCRModule(OCRAbstact):
    """OCR module using the EasyOCR engine."""

    def __init__(self, config: Union[EasyOCRConfig, dict]) -> None:
        """Initialize the EasyOCR reader.

        Args:
            config: An EasyOCRConfig instance or a dict that will be
                unpacked into one.
        """
        if isinstance(config, dict):
            config = EasyOCRConfig(**config)
        super().__init__(config)
        self._reader = easyocr.Reader(self.config.languages)

    def _run_single(self, image: np.ndarray, model_params: dict) -> OCRResult:
        """Run EasyOCR on a single image.

        Args:
            image: Input image as a numpy array (H x W x C).
            model_params: Additional keyword arguments passed to
                ``easyocr.Reader.readtext``.

        Returns:
            An OCRResult containing detected text regions.
        """
        results = self._reader.readtext(image, **model_params)
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
        return OCRResult(bounding_boxes=bboxes)
