import pytest

from ocr_backbone.bounding_box import BoundingBox


def test_bounding_box_creation():
    bbox = BoundingBox(
        coordinates=((0, 0), (10, 10)),
        text="hello",
        confidence=0.95,
    )
    assert bbox.text == "hello"
    assert bbox.confidence == 0.95
    assert bbox.coordinates == ((0, 0), (10, 10))


def test_bounding_box_default_confidence():
    bbox = BoundingBox(coordinates=((0, 0), (5, 5)), text="test")
    assert bbox.confidence == 0.0


def test_bounding_box_equality():
    bbox1 = BoundingBox(((0, 0), (1, 1)), "a", 0.5)
    bbox2 = BoundingBox(((0, 0), (1, 1)), "a", 0.5)
    assert bbox1 == bbox2


def test_bounding_box_invalid_coordinate_count():
    with pytest.raises(ValueError, match="exactly two"):
        BoundingBox(coordinates=((0, 0),), text="bad")


def test_bounding_box_non_integer_coordinates():
    with pytest.raises(ValueError, match="integers"):
        BoundingBox(coordinates=((0.5, 0), (5, 5)), text="bad")


def test_bounding_box_top_left_greater_than_bottom_right():
    with pytest.raises(ValueError, match="above and to the left"):
        BoundingBox(coordinates=((10, 0), (5, 5)), text="bad")
    with pytest.raises(ValueError, match="above and to the left"):
        BoundingBox(coordinates=((0, 10), (5, 5)), text="bad")
