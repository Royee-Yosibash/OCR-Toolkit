import unittest

from ocr_backbone.bounding_box import BoundingBox


class TestBoundingBox(unittest.TestCase):

    def test_valid_creation(self):
        bbox = BoundingBox(
            coordinates=((0, 0), (10, 10)),
            text="hello",
            confidence=0.95,
        )
        self.assertEqual(bbox.text, "hello")
        self.assertEqual(bbox.confidence, 0.95)
        self.assertEqual(bbox.coordinates, ((0, 0), (10, 10)))

    def test_equality(self):
        bbox1 = BoundingBox(((0, 0), (1, 1)), "a", 0.5)
        bbox2 = BoundingBox(((0, 0), (1, 1)), "a", 0.5)
        self.assertEqual(bbox1, bbox2)

    def test_from_dict(self):
        data = {"coordinates": [[0, 0], [10, 10]], "text": "hi", "confidence": 0.9}
        bbox = BoundingBox.from_dict(data)
        self.assertEqual(bbox.coordinates, ((0, 0), (10, 10)))
        self.assertEqual(bbox.text, "hi")
        self.assertEqual(bbox.confidence, 0.9)

    def test_from_dict_default_confidence(self):
        data = {"coordinates": [[0, 0], [5, 5]], "text": "no conf"}
        bbox = BoundingBox.from_dict(data)
        self.assertEqual(bbox.confidence, 0.0)

    def test_dict_roundtrip(self):
        original = BoundingBox(coordinates=((1, 2), (30, 40)), text="test", confidence=0.75)
        restored = BoundingBox.from_dict(original.to_dict())
        self.assertEqual(original, restored)

    def test_to_pixel_list(self):
        bbox = BoundingBox(coordinates=((1, 2), (3, 4)), text="t")
        pixels = bbox.to_pixel_list()
        self.assertEqual(len(pixels), 9)
        self.assertEqual(pixels[0], (1, 2))
        self.assertEqual(pixels[-1], (3, 4))
        self.assertIn((2, 3), pixels)

    def test_pixel_list_roundtrip(self):
        original = BoundingBox(coordinates=((2, 3), (7, 9)), text="round", confidence=0.85)
        pixels = original.to_pixel_list()
        restored = BoundingBox.from_pixel_list(pixels, text=original.text, confidence=original.confidence)
        self.assertEqual(original, restored)

    def test_from_pixel_list_empty(self):
        with self.assertRaises(ValueError):
            BoundingBox.from_pixel_list([])

    def test_to_pixel_list_single_pixel(self):
        bbox = BoundingBox(coordinates=((5, 5), (5, 5)), text="dot")
        self.assertEqual(bbox.to_pixel_list(), [(5, 5)])

    def test_lt_by_y_then_x(self):
        a = BoundingBox(coordinates=((0, 0), (5, 5)), text="a")
        b = BoundingBox(coordinates=((0, 10), (5, 15)), text="b")
        c = BoundingBox(coordinates=((10, 0), (15, 5)), text="c")
        self.assertTrue(a < b)
        self.assertTrue(a < c)
        self.assertTrue(c < b)

    def test_sort_order(self):
        a = BoundingBox(coordinates=((10, 0), (20, 10)), text="a")
        b = BoundingBox(coordinates=((0, 5), (10, 15)), text="b")
        c = BoundingBox(coordinates=((0, 0), (5, 5)), text="c")
        result = sorted([a, b, c])
        self.assertEqual([bb.text for bb in result], ["c", "a", "b"])

    def test_non_integer_coordinates(self):
        with self.assertRaises(ValueError):
            BoundingBox(coordinates=((0.5, 0), (5, 5)), text="bad")

    def test_top_left_greater_than_bottom_right(self):
        with self.assertRaises(ValueError):
            BoundingBox(coordinates=((10, 0), (5, 5)), text="bad")
        with self.assertRaises(ValueError):
            BoundingBox(coordinates=((0, 10), (5, 5)), text="bad")
