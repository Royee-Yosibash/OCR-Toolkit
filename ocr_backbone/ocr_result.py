from dataclasses import dataclass, field

from ocr_backbone.polygon import Polygon
from utils.serialize_utils import SerializableClass


@dataclass
class OCRResult(SerializableClass):
    """Output of an OCR run on a single image.

    Args:
        detections: List of detected text regions. Each entry is either a
            Polygon (3+ vertices) or a BoundingBox (exactly 2 vertices).
    """

    detections: list[Polygon] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Sort detections in reading order after initialization."""
        self.detections.sort()

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
        if len(self.detections) != len(other.detections):
            return False
        for a, b in zip(self.detections, other.detections, strict=True):
            if a.coordinates != b.coordinates:
                return False
            if a.text != b.text:
                return False
            if abs(a.confidence - b.confidence) > confidence_tolerance:
                return False
        return True
