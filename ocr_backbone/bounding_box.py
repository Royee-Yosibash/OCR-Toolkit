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

    def __lt__(self, other: object) -> bool:
        """Compare bounding boxes by top-left position (y then x).

        Args:
            other: Another BoundingBox to compare against.

        Returns:
            True if this bounding box comes before ``other`` in
            top-to-bottom, left-to-right reading order.
        """
        if not isinstance(other, BoundingBox):
            return NotImplemented
        self_tl = self.coordinates[0]
        other_tl = other.coordinates[0]
        return (self_tl[1], self_tl[0]) < (other_tl[1], other_tl[0])

    def _remap_bounding_box(self, x_offset: int, y_offset: int):
        """Remap a bounding box from sub-image coordinates to original image coordinates.

        Args:
            x_offset: Horizontal pixel offset of the sub-image in the original image.
            y_offset: Vertical pixel offset of the sub-image in the original image.
        """
        top_left, bottom_right = self.coordinates
        new_coords = (
            (top_left[0] + x_offset, top_left[1] + y_offset),
            (bottom_right[0] + x_offset, bottom_right[1] + y_offset),
        )
        self.coordinates = new_coords

    @classmethod
    def from_dict(cls, data: dict):
        """Create a BoundingBox from a dict produced by ``to_dict``.

        Args:
            data: A dict with coordinates, text, and optionally confidence.

        Returns:
            A BoundingBox instance.
        """
        coords = data["coordinates"]
        return cls(
            coordinates=(tuple(coords[0]), tuple(coords[1])),
            text=data["text"],
            confidence=data.get("confidence", 0.0),
        )

    @classmethod
    def from_pixel_list(
        cls, pixels: list[tuple[int, int]], text: str = "", confidence: float = 0.0
        ):
        """Create a BoundingBox from a list of pixel coordinates.

        Computes the axis-aligned bounding box that encloses all given pixels.

        Args:
            pixels: A non-empty list of (x, y) pixel coordinates.
            text: The text associated with the bounding box.
            confidence: Confidence score of the detection.

        Returns:
            A BoundingBox enclosing all provided pixels.

        Raises:
            ValueError: If the pixel list is empty.
        """
        if not pixels:
            raise ValueError("Pixel list must not be empty.")
        xs, ys = zip(*pixels)
        return cls(
            coordinates=((min(xs), min(ys)), (max(xs), max(ys))),
            text=text,
            confidence=confidence,
        )

    def to_dict(self) -> dict:
        """Convert the bounding box to a JSON-serializable dict.

        Returns:
            A dict with coordinates, text, and confidence fields.
        """
        return {
            "coordinates": [list(self.coordinates[0]), list(self.coordinates[1])],
            "text": self.text,
            "confidence": round(self.confidence, 6),
        }

    def to_pixel_list(self) -> list[tuple[int, int]]:
        """Return all pixel coordinates contained in the bounding box.

        Returns:
            A list of (x, y) tuples for every pixel from top_left to
            bottom_right (inclusive).
        """
        top_left, bottom_right = self.coordinates
        return [
            (x, y)
            for y in range(top_left[1], bottom_right[1] + 1)
            for x in range(top_left[0], bottom_right[0] + 1)
        ]
