from dataclasses import dataclass

from utils.serialize_utils import SerializableClass


@dataclass
class Polygon(SerializableClass):
    """Represents a detected text region described by an arbitrary polygon.

    Args:
        coordinates: A tuple of (x, y) integer pairs defining the boundary of
            the polygon in order. Must contain at least 3 points.
        text: The recognized text within the polygon.
        confidence: Confidence score of the detection, in range [0, 1].

    Raises:
        ValueError: If coordinates contains fewer than 3 points, or if any
            coordinate value is not an integer.
    """

    coordinates: tuple[tuple[int, int], ...]
    text: str
    confidence: float = 0.0

    def __post_init__(self) -> None:
        """Validates that coordinates are well-formed."""
        if len(self.coordinates) < 3:
            raise ValueError("A polygon must have at least 3 vertices.")
        if any(len(point) != 2 for point in self.coordinates):
            raise ValueError("Each vertex must be an (x, y) pair.")
        all_values = [v for point in self.coordinates for v in point]
        if not all(isinstance(v, int) for v in all_values):
            raise ValueError("All vertex coordinate values must be integers.")

    def _top_left_vertex(self) -> tuple[int, int]:
        """Return the topmost vertex, breaking ties by leftmost.

        Returns:
            The (x, y) pair with the smallest y, then smallest x.
        """
        return min(self.coordinates, key=lambda v: (v[1], v[0]))

    def __lt__(self, other: object) -> bool:
        """Compare polygons by their topmost (then leftmost) vertex.

        Args:
            other: Another Polygon to compare against.

        Returns:
            True if this polygon's topmost vertex comes before the
            other's in top-to-bottom, left-to-right reading order.
        """
        if not isinstance(other, Polygon):
            return NotImplemented
        sy, sx = self._top_left_vertex()[1], self._top_left_vertex()[0]
        oy, ox = other._top_left_vertex()[1], other._top_left_vertex()[0]
        return (sy, sx) < (oy, ox)

    def remap_coordinates(self, x_offset: int, y_offset: int) -> None:
        """Translate all vertices by the given offsets.

        Args:
            x_offset: Horizontal pixel offset to add.
            y_offset: Vertical pixel offset to add.
        """
        self.coordinates = tuple((x + x_offset, y + y_offset) for x, y in self.coordinates)

    def to_dict(self) -> dict:
        raw_dict = super().to_dict()
        raw_dict["confidence"] = round(self.confidence, 6)
        return raw_dict
