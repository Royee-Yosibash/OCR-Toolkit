from __future__ import annotations

from dataclasses import dataclass, field

from ocr_backbone.bounding_box import BoundingBox


@dataclass
class OCRResult:
    """Output of an OCR run on a single image.

    Args:
        bounding_boxes: List of detected text regions.
    """

    bounding_boxes: list[BoundingBox] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Sort bounding boxes in reading order after initialization."""
        self.bounding_boxes = [BoundingBox.from_dict(bb) if isinstance(bb, dict) else bb
                               for bb in self.bounding_boxes]
        self.bounding_boxes.sort()

    def to_dict(self) -> dict:
        """Convert the result to a JSON-serializable dict.

        Returns:
            A dict with a bounding_boxes field containing a list of BB dicts.
        """
        return {
            "bounding_boxes": [bb.to_dict() for bb in self.bounding_boxes],
        }

    @classmethod
    def from_dict(cls, data: dict) -> OCRResult:
        """Create an OCRResult from a dict produced by ``to_dict``.

        Args:
            data: A dict with a bounding_boxes field.

        Returns:
            An OCRResult instance.
        """
        return cls(**data)

    def is_close(self, other: OCRResult, confidence_tolerance: float = 1e-3) -> bool:
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
        for a, b in zip(self.bounding_boxes, other.bounding_boxes):
            if a.coordinates != b.coordinates:
                return False
            if a.text != b.text:
                return False
            if abs(a.confidence - b.confidence) > confidence_tolerance:
                return False
        return True
