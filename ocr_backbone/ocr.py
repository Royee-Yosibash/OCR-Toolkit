from abc import ABC, abstractmethod

import numpy as np

from ocr_backbone.bounding_box import BoundingBox


class OCRAbstact(ABC):
    """Abstract base class for OCR engines.

    Subclasses must implement ``__init__`` to set up the underlying engine
    and ``run`` to perform text detection on an image.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the OCR engine."""

    @abstractmethod
    def get_text_bb(self, image: np.ndarray, config: dict) -> list[BoundingBox]:
        """Run OCR on an image and return detected text regions.

        Args:
            image: Input image as a numpy array (H x W x C).
            config: The ocr configurations parameters.

        Returns:
            A list of BoundingBox instances for each detected text region.
        """
