import unittest

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr_result import OCRResult


class TestOCRResult(unittest.TestCase):

    def test_default_empty(self):
        result = OCRResult()
        self.assertEqual(result.bounding_boxes, [])

    def test_to_dict(self):
        result = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
        ])
        d = result.to_dict()
        self.assertEqual(len(d["bounding_boxes"]), 1)
        self.assertEqual(d["bounding_boxes"][0]["text"], "hi")

    def test_from_dict(self):
        data = {
            "bounding_boxes": [
                {"coordinates": [[0, 0], [5, 5]], "text": "a", "confidence": 0.5},
                {"coordinates": [[1, 1], [3, 3]], "text": "b"},
            ]
        }
        result = OCRResult.from_dict(data)
        self.assertEqual(len(result.bounding_boxes), 2)
        self.assertEqual(result.bounding_boxes[0].text, "a")
        self.assertEqual(result.bounding_boxes[1].confidence, 0.0)

    def test_roundtrip(self):
        original = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((1, 2), (10, 20)), text="x", confidence=0.75),
            BoundingBox(coordinates=((3, 4), (30, 40)), text="y", confidence=0.5),
        ])
        restored = OCRResult.from_dict(original.to_dict())
        self.assertEqual(original, restored)

    def test_from_dict_missing_key(self):
        result = OCRResult.from_dict({})
        self.assertEqual(result.bounding_boxes, [])

    def test_auto_sorts_on_init(self):
        b = BoundingBox(coordinates=((0, 10), (5, 15)), text="b")
        a = BoundingBox(coordinates=((0, 0), (5, 5)), text="a")
        result = OCRResult(bounding_boxes=[b, a])
        self.assertEqual(result.bounding_boxes[0].text, "a")
        self.assertEqual(result.bounding_boxes[1].text, "b")

    def test_from_dict_sorts(self):
        data = {
            "bounding_boxes": [
                {"coordinates": [[0, 10], [5, 15]], "text": "second"},
                {"coordinates": [[0, 0], [5, 5]], "text": "first"},
            ]
        }
        result = OCRResult.from_dict(data)
        self.assertEqual(result.bounding_boxes[0].text, "first")

    def test_is_close_exact(self):
        a = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
        ])
        b = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
        ])
        self.assertTrue(a.is_close(b))

    def test_is_close_within_tolerance(self):
        a = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
        ])
        b = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9005),
        ])
        self.assertTrue(a.is_close(b))

    def test_is_close_exceeds_tolerance(self):
        a = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
        ])
        b = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.5),
        ])
        self.assertFalse(a.is_close(b))

    def test_is_close_different_text(self):
        a = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
        ])
        b = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="bye", confidence=0.9),
        ])
        self.assertFalse(a.is_close(b))

    def test_is_close_different_count(self):
        a = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
        ])
        b = OCRResult(bounding_boxes=[])
        self.assertFalse(a.is_close(b))
