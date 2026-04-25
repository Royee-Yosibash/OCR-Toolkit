import unittest

from evaluation.ocr_ground_truth import OCRGroundTruth
from ocr_backbone.bounding_box import BoundingBox


class TestOCRGroundTruth(unittest.TestCase):
    def test_default_empty(self):
        gt = OCRGroundTruth()
        self.assertEqual(gt.detections, [])
        self.assertEqual(gt.tags, [])

    def test_to_dict_includes_tags(self):
        gt = OCRGroundTruth(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
            ],
            tags=["handwritten", "english"],
        )
        d = gt.to_dict()
        self.assertEqual(d["tags"], ["handwritten", "english"])
        self.assertEqual(len(d["detections"]), 1)

    def test_from_dict_with_tags(self):
        data = {
            "detections": [
                {"_type": "BoundingBox", "coordinates": [[0, 0], [5, 5]], "text": "a", "confidence": 0.5},
            ],
            "tags": ["printed"],
        }
        gt = OCRGroundTruth.from_dict(data)
        self.assertIsInstance(gt, OCRGroundTruth)
        self.assertEqual(gt.tags, ["printed"])
        self.assertEqual(len(gt.detections), 1)

    def test_from_dict_missing_tags(self):
        data = {
            "detections": [
                {"_type": "BoundingBox", "coordinates": [[0, 0], [5, 5]], "text": "a", "confidence": 0.5},
            ],
        }
        gt = OCRGroundTruth.from_dict(data)
        self.assertEqual(gt.tags, [])

    def test_roundtrip(self):
        original = OCRGroundTruth(
            detections=[
                BoundingBox(coordinates=((1, 2), (10, 20)), text="x", confidence=0.75),
            ],
            tags=["tag1", "tag2"],
        )
        restored = OCRGroundTruth.from_dict(original.to_dict())
        self.assertEqual(original, restored)

    def test_is_close_same_tags(self):
        a = OCRGroundTruth(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
            ],
            tags=["a"],
        )
        b = OCRGroundTruth(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
            ],
            tags=["a"],
        )
        self.assertTrue(a.is_close(b))

    def test_is_close_different_tags(self):
        a = OCRGroundTruth(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
            ],
            tags=["a"],
        )
        b = OCRGroundTruth(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
            ],
            tags=["b"],
        )
        self.assertFalse(a.is_close(b))

    def test_tags_lowercased_on_init(self):
        gt = OCRGroundTruth(tags=["Printed", "ENGLISH", "Mixed"])
        self.assertEqual(gt.tags, ["printed", "english", "mixed"])

    def test_tags_lowercased_from_dict(self):
        data = {
            "detections": [],
            "tags": ["Handwritten", "FRENCH"],
        }
        gt = OCRGroundTruth.from_dict(data)
        self.assertEqual(gt.tags, ["handwritten", "french"])

    def test_is_close_bboxes_differ(self):
        a = OCRGroundTruth(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hi", confidence=0.9),
            ],
            tags=["a"],
        )
        b = OCRGroundTruth(
            detections=[],
            tags=["a"],
        )
        self.assertFalse(a.is_close(b))
