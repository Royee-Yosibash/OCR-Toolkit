from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields

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

    def to_dict(self) -> dict:
        """Convert the ground truth to a JSON-serializable dict.

        Returns:
            A dict with bounding_boxes and tags fields.
        """
        data = super().to_dict()
        data["tags"] = self.tags
        return data

    @classmethod
    def from_dict(cls, data: dict) -> OCRGroundTruth:
        """Create an OCRGroundTruth from a dict produced by ``to_dict``.

        Args:
            data: A dict with bounding_boxes and optionally tags fields.

        Returns:
            An OCRGroundTruth instance.
        """
        # TODO: change this when we handle nested from_dict()
        base = super().from_dict(data)
        kwargs = {f.name: getattr(base, f.name) for f in dataclass_fields(base)}
        kwargs["tags"] = data.get("tags", [])
        return cls(**kwargs)

    def is_close(self, other: OCRGroundTruth, confidence_tolerance: float = 1e-3) -> bool:
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
