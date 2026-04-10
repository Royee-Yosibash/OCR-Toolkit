from dataclasses import dataclass, field

from ocr_backbone.bounding_box import BoundingBox
from utils.serialize_utils import SerializableClass


@dataclass
class OCRResult(SerializableClass):
    """Output of an OCR run on a single image.

    Args:
        bounding_boxes: List of detected text regions.
    """

    bounding_boxes: list[BoundingBox] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Sort bounding boxes in reading order after initialization."""
        self.bounding_boxes.sort()

    def is_close(self, other: "OCRResult", confidence_tolerance: float = 1e-3) -> bool:
        """Check if two OCRResults are equivalent within a confidence tolerance.

        Coordinates and text must match exactly. Confidence values may differ
        by up to ``confidence_tolerance``.

        Args:
            other: The OCRResult to compare against.
            confidence_tolerance: Maximum allowed difference in confidence.

        Returns:
            True if the results match within the tolerance.
        """
        if len(self.bounding_boxes) != len(other.bounding_boxes):
            return False
        for a, b in zip(self.bounding_boxes, other.bounding_boxes, strict=True):
            if a.coordinates != b.coordinates:
                return False
            if a.text != b.text:
                return False
            if abs(a.confidence - b.confidence) > confidence_tolerance:
                return False
        return True
