import unittest

from ocr_backbone.bounding_box import BoundingBox
from evaluation.ocr_ground_truth import OCRGroundTruth


class TestOCRGroundTruth(unittest.TestCase):

    def test_default_empty(self):
        gt = OCRGroundTruth()
        self.assertEqual(gt.bounding_boxes, [])
        self.assertEqual(gt.tags, [])

    def test_to_dict_includes_tags(self):
        gt = OCRGroundTruth(
            bounding_boxes=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
            ],
            tags=["handwritten", "english"],
        )
        d = gt.to_dict()
        self.assertEqual(d["tags"], ["handwritten", "english"])
        self.assertEqual(len(d["bounding_boxes"]), 1)

    def test_from_dict_with_tags(self):
        data = {
            "bounding_boxes": [
                {"coordinates": [[0, 0], [5, 5]], "text": "a", "confidence": 0.5},
            ],
            "tags": ["printed"],
        }
        gt = OCRGroundTruth.from_dict(data)
        self.assertIsInstance(gt, OCRGroundTruth)
        self.assertEqual(gt.tags, ["printed"])
        self.assertEqual(len(gt.bounding_boxes), 1)

    def test_from_dict_missing_tags(self):
        data = {
            "bounding_boxes": [
                {"coordinates": [[0, 0], [5, 5]], "text": "a", "confidence": 0.5},
            ],
        }
        gt = OCRGroundTruth.from_dict(data)
        self.assertEqual(gt.tags, [])

    def test_roundtrip(self):
        original = OCRGroundTruth(
            bounding_boxes=[
                BoundingBox(coordinates=((1, 2), (10, 20)), text="x", confidence=0.75),
            ],
            tags=["tag1", "tag2"],
        )
        restored = OCRGroundTruth.from_dict(original.to_dict())
        self.assertEqual(original, restored)

    def test_is_close_same_tags(self):
        a = OCRGroundTruth(
            bounding_boxes=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
            ],
            tags=["a"],
        )
        b = OCRGroundTruth(
            bounding_boxes=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
            ],
            tags=["a"],
        )
        self.assertTrue(a.is_close(b))

    def test_is_close_different_tags(self):
        a = OCRGroundTruth(
            bounding_boxes=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
            ],
            tags=["a"],
        )
        b = OCRGroundTruth(
            bounding_boxes=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
            ],
            tags=["b"],
        )
        self.assertFalse(a.is_close(b))

    def test_is_close_bboxes_differ(self):
        a = OCRGroundTruth(
            bounding_boxes=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
            ],
            tags=["a"],
        )
        b = OCRGroundTruth(
            bounding_boxes=[],
            tags=["a"],
        )
        self.assertFalse(a.is_close(b))
