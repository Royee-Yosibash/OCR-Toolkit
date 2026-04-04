from dataclasses import dataclass, field

from ocr_backbone.ocr_result import OCRResult


@dataclass
class OCRGroundTruth(OCRResult):
    """Ground truth OCR result that also carries semantic tags.

    Args:
        tags: List of string tags associated with the ground truth.
    """

    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Sort bounding boxes and normalize tags to lowercase."""
        super().__post_init__()
        self.tags = [t.lower() for t in self.tags]

    def is_close(self, other: 'OCRGroundTruth', confidence_tolerance: float = 1e-3) -> bool:
        """Also compares tags on top of the base bounding box comparison.

        Args:
            other: The OCRGroundTruth to compare against.
            confidence_tolerance: Maximum allowed difference in confidence.

        Returns:
            True if the results and tags match within the tolerance.
        """
        if not super().is_close(other, confidence_tolerance):
            return False
        return set(self.tags) == set(other.tags)
