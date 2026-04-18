from dataclasses import dataclass

from ocr_backbone.polygon import Polygon


@dataclass
class BoundingBox(Polygon):
    """Represents a detected text region as an axis-aligned rectangle.

    A specialization of Polygon constrained to exactly two (x, y) pairs
    representing the top-left and bottom-right corners.

    Args:
        coordinates: A tuple of two (x, y) integer pairs: (top_left,
            bottom_right), where top_left is the upper-left corner and
            bottom_right is the lower-right corner. Values represent pixel
            positions and must be integers. top_left values must be less
            than or equal to the corresponding bottom_right values.

    Raises:
        ValueError: If coordinates does not contain exactly two points, if
            top_left is not above and to the left of bottom_right, or if
            any coordinate value is not an integer.
    """

    coordinates: tuple[tuple[int, int], tuple[int, int]]

    def __post_init__(self) -> None:
        """Validates that coordinates form a proper axis-aligned rectangle."""
        if len(self.coordinates) != 2 or any(len(point) != 2 for point in self.coordinates):
            raise ValueError("coordinates must contain exactly two (x, y) pairs: (top_left, bottom_right).")
        all_values = [v for point in self.coordinates for v in point]
        if not all(isinstance(v, int) for v in all_values):
            raise ValueError("All coordinate values must be integers.")
        top_left, bottom_right = self.coordinates
        if top_left[0] > bottom_right[0] or top_left[1] > bottom_right[1]:
            raise ValueError(f"top_left {top_left} must be above and to the left of bottom_right {bottom_right}.")

    def _remap_bounding_box(self, x_offset: int, y_offset: int):
        """Remap a bounding box from sub-image coordinates to original image coordinates.

        Args:
            x_offset: Horizontal pixel offset of the sub-image in the original image.
            y_offset: Vertical pixel offset of the sub-image in the original image.
        """
        top_left, bottom_right = self.coordinates
        self.coordinates = (
            (top_left[0] + x_offset, top_left[1] + y_offset),
            (bottom_right[0] + x_offset, bottom_right[1] + y_offset),
        )

    @classmethod
    def from_pixel_list(cls, pixels: list[tuple[int, int]], text: str = "", confidence: float = 0.0):
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
        xs, ys = zip(*pixels, strict=True)
        return cls(
            coordinates=((min(xs), min(ys)), (max(xs), max(ys))),
            text=text,
            confidence=confidence,
        )

    def to_pixel_list(self) -> list[tuple[int, int]]:
        """Return all pixel coordinates contained in the bounding box.

        Returns:
            A list of (x, y) tuples for every pixel from top_left to
            bottom_right (inclusive).
        """
        top_left, bottom_right = self.coordinates
        return [
            (x, y) for y in range(top_left[1], bottom_right[1] + 1) for x in range(top_left[0], bottom_right[0] + 1)
        ]
