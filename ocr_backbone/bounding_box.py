from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Represents a detected text region from an OCR engine.

    Args:
        coordinates: A tuple of two (x, y) integer pairs: (top_left,
            bottom_right), where top_left is the upper-left corner and
            bottom_right is the lower-right corner. Values represent pixel
            positions and must be integers. top_left values must be less
            than or equal to the corresponding bottom_right values.
        text: The recognized text within the bounding box.
        confidence: Confidence score of the detection, in range [0, 1].

    Raises:
        ValueError: If coordinates does not contain exactly two points, if
            top_left is not above and to the left of bottom_right, or if
            any coordinate value is not an integer.
    """

    coordinates: tuple[tuple[int, int], tuple[int, int]]
    text: str
    confidence: float = 0.0

    def __post_init__(self) -> None:
        """Validates that coordinates are well-formed."""
        if len(self.coordinates) != 2 or any(
            len(point) != 2 for point in self.coordinates
        ):
            raise ValueError(
                "coordinates must contain exactly two (x, y) pairs: "
                "(top_left, bottom_right)."
            )
        all_values = [v for point in self.coordinates for v in point]
        if not all(isinstance(v, int) for v in all_values):
            raise ValueError("All coordinate values must be integers.")
        top_left, bottom_right = self.coordinates
        if top_left[0] > bottom_right[0] or top_left[1] > bottom_right[1]:
            raise ValueError(
                f"top_left {top_left} must be above and to the left of "
                f"bottom_right {bottom_right}."
            )
