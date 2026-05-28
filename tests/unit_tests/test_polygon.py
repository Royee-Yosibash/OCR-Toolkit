import unittest

from ocr_backbone.polygon import Polygon


class TestPolygon(unittest.TestCase):
    """Tests for the ``Polygon`` class: construction, defaults, and coordinate handling."""

    def test_valid_creation(self):
        poly = Polygon(
            coordinates=((0, 0), (10, 0), (10, 10)),
            text="hello",
            confidence=0.95,
        )
        self.assertEqual(poly.text, "hello")
        self.assertEqual(poly.confidence, 0.95)
        self.assertEqual(poly.coordinates, ((0, 0), (10, 0), (10, 10)))

    def test_default_confidence(self):
        poly = Polygon(coordinates=((0, 0), (1, 0), (1, 1)), text="x")
        self.assertEqual(poly.confidence, 0.0)

    def test_equality(self):
        poly1 = Polygon(((0, 0), (1, 0), (1, 1)), "a", 0.5)
        poly2 = Polygon(((0, 0), (1, 0), (1, 1)), "a", 0.5)
        self.assertEqual(poly1, poly2)

    def test_fewer_than_three_vertices_raises(self):
        with self.assertRaises(ValueError):
            Polygon(coordinates=((0, 0), (1, 1)), text="bad")

    def test_non_integer_coordinates_raises(self):
        with self.assertRaises(ValueError):
            Polygon(coordinates=((0.5, 0), (5, 0), (5, 5)), text="bad")

    def test_malformed_vertex_raises(self):
        with self.assertRaises(ValueError):
            Polygon(coordinates=((0, 0, 0), (1, 0), (1, 1)), text="bad")

    def test_to_dict(self):
        poly = Polygon(
            coordinates=((1, 2), (30, 2), (30, 40)),
            text="test",
            confidence=0.7512345,
        )
        d = poly.to_dict()
        self.assertEqual(d["coordinates"], ((1, 2), (30, 2), (30, 40)))
        self.assertEqual(d["text"], "test")
        self.assertEqual(d["confidence"], 0.751235)

    def test_dict_roundtrip(self):
        original = Polygon(
            coordinates=((1, 2), (30, 2), (30, 40)),
            text="test",
            confidence=0.75,
        )
        restored = Polygon.from_dict(original.to_dict())
        self.assertEqual(original, restored)

    def test_lt_topmost_wins(self):
        a = Polygon(coordinates=((5, 0), (10, 5), (0, 5)), text="a")
        b = Polygon(coordinates=((0, 10), (10, 10), (5, 20)), text="b")
        self.assertTrue(a < b)
        self.assertFalse(b < a)

    def test_lt_same_y_leftmost_wins(self):
        a = Polygon(coordinates=((0, 0), (5, 5), (0, 5)), text="a")
        b = Polygon(coordinates=((3, 0), (8, 5), (3, 5)), text="b")
        self.assertTrue(a < b)
        self.assertFalse(b < a)

    def test_lt_uses_topmost_vertex_not_first(self):
        a = Polygon(coordinates=((10, 20), (20, 20), (15, 1)), text="a")
        b = Polygon(coordinates=((0, 5), (10, 5), (5, 15)), text="b")
        self.assertTrue(a < b)

    def test_lt_not_implemented_for_other_types(self):
        poly = Polygon(coordinates=((0, 0), (1, 0), (1, 1)), text="a")
        self.assertEqual(poly.__lt__("not a polygon"), NotImplemented)

    def test_translate_coordinates_positive(self):
        poly = Polygon(coordinates=((0, 0), (5, 0), (5, 5)), text="t", confidence=0.9)
        translated = poly.translate_coordinates(3, 4)
        self.assertEqual(translated.coordinates, ((3, 4), (8, 4), (8, 9)))
        self.assertEqual(translated.text, "t")
        self.assertEqual(translated.confidence, 0.9)

    def test_translate_coordinates_negative_raises(self):
        poly = Polygon(coordinates=((0, 0), (5, 0), (5, 5)), text="t")
        with self.assertRaises(ValueError):
            poly.translate_coordinates(-10, 0)

    def test_sort_order(self):
        a = Polygon(coordinates=((10, 0), (20, 0), (15, 10)), text="a")
        b = Polygon(coordinates=((0, 5), (10, 5), (5, 15)), text="b")
        c = Polygon(coordinates=((0, 0), (5, 0), (2, 5)), text="c")
        result = sorted([b, a, c])
        self.assertEqual([p.text for p in result], ["c", "a", "b"])
