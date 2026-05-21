from collections import Counter
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
        """Sort bounding boxes, normalize tags to lowercase, and reject duplicates.

        Raises:
            ValueError: If any tag appears more than once after
                case-insensitive normalization.
        """
        super().__post_init__()
        self.tags = [t.lower() for t in self.tags]
        duplicates = sorted(t for t, count in Counter(self.tags).items() if count > 1)
        if duplicates:
            raise ValueError(f"Duplicate tag(s) after case-insensitive normalization: {duplicates}")

    def is_close(self, other: "OCRGroundTruth", confidence_tolerance: float = 1e-3) -> bool:
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
