import unittest

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr_result import OCRResult
from evaluation.metrics import (
    OCRResultCER,
    OCRResultWER,
    word_count_ratio,
    word_recall,
    word_precision,
)


def _make_result(text: str) -> OCRResult:
    """Create an OCRResult with a single bounding box from a text string.

    Args:
        text: The text content for the bounding box.

    Returns:
        An OCRResult containing one bounding box with the given text,
        or an empty OCRResult if text is empty.
    """
    if not text:
        return OCRResult()
    return OCRResult(bounding_boxes=[
        BoundingBox(coordinates=((0, 0), (10, 10)), text=text),
    ])


class TestOCRResultCER(unittest.TestCase):
    """Tests for OCRResultCER."""

    def setUp(self):
        self.metric = OCRResultCER()

    def test_identical(self):
        self.assertAlmostEqual(self.metric(_make_result("hello"), _make_result("hello")), 0.0)

    def test_empty_both(self):
        self.assertAlmostEqual(self.metric(_make_result(""), _make_result("")), 0.0)

    def test_empty_ground_truth(self):
        self.assertEqual(self.metric(_make_result("abc"), _make_result("")), 3.0)

    def test_empty_prediction(self):
        self.assertAlmostEqual(self.metric(_make_result(""), _make_result("abc")), 1.0)

    def test_substitution(self):
        self.assertAlmostEqual(self.metric(_make_result("abc"), _make_result("axc")), 1 / 3)

    def test_insertion(self):
        self.assertAlmostEqual(self.metric(_make_result("abcd"), _make_result("abc")), 1 / 3)

    def test_is_unbounded(self):
        self.assertFalse(self.metric.is_bounded)


class TestOCRResultWER(unittest.TestCase):
    """Tests for OCRResultWER."""

    def setUp(self):
        self.metric = OCRResultWER()

    def test_identical(self):
        self.assertAlmostEqual(self.metric(_make_result("hello world"), _make_result("hello world")), 0.0)

    def test_empty_both(self):
        self.assertAlmostEqual(self.metric(_make_result(""), _make_result("")), 0.0)

    def test_empty_ground_truth(self):
        self.assertEqual(self.metric(_make_result("hello world"), _make_result("")), 2.0)

    def test_one_substitution(self):
        self.assertAlmostEqual(self.metric(_make_result("hello earth"), _make_result("hello world")), 0.5)

    def test_completely_wrong(self):
        self.assertAlmostEqual(self.metric(_make_result("foo bar"), _make_result("hello world")), 1.0)

    def test_is_unbounded(self):
        self.assertFalse(self.metric.is_bounded)

    def test_split_bbs_equivalent(self):
        one_bb = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (20, 10)), text="hello world"),
        ])
        two_bbs = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello"),
            BoundingBox(coordinates=((10, 0), (20, 10)), text="world"),
        ])
        self.assertAlmostEqual(self.metric(two_bbs, one_bb), 0.0)

    def test_special_chars_ignored(self):
        pred = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello\nworld"),
        ])
        gt = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world"),
        ])
        self.assertAlmostEqual(self.metric(pred, gt), 0.0)

    def test_punctuation_ignored(self):
        pred = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello, world."),
        ])
        gt = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world"),
        ])
        self.assertAlmostEqual(self.metric(pred, gt), 0.0)


class TestWordCountRatio(unittest.TestCase):
    """Tests for word_count_ratio."""

    def test_equal_counts(self):
        pred = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world"),
        ])
        gt = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="foo bar"),
        ])
        self.assertAlmostEqual(word_count_ratio(pred, gt), 1.0)

    def test_over_detection(self):
        pred = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="a b c"),
        ])
        gt = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="a b"),
        ])
        self.assertAlmostEqual(word_count_ratio(pred, gt), 1.5)

    def test_both_empty(self):
        self.assertAlmostEqual(word_count_ratio(OCRResult(), OCRResult()), 1.0)


class TestWordRecall(unittest.TestCase):
    """Tests for word_recall."""

    def test_perfect(self):
        bbs = [BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world")]
        self.assertAlmostEqual(
            word_recall(OCRResult(bounding_boxes=list(bbs)), OCRResult(bounding_boxes=list(bbs))),
            1.0,
        )

    def test_partial(self):
        pred = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello"),
        ])
        gt = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world"),
        ])
        self.assertAlmostEqual(word_recall(pred, gt), 0.5)

    def test_split_bbs(self):
        pred = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello"),
            BoundingBox(coordinates=((10, 0), (20, 10)), text="world"),
        ])
        gt = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (20, 10)), text="hello world"),
        ])
        self.assertAlmostEqual(word_recall(pred, gt), 1.0)

    def test_both_empty(self):
        self.assertAlmostEqual(word_recall(OCRResult(), OCRResult()), 1.0)

    def test_duplicate_words(self):
        pred = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello"),
        ])
        gt = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello hello"),
        ])
        self.assertAlmostEqual(word_recall(pred, gt), 0.5)


class TestWordPrecision(unittest.TestCase):
    """Tests for word_precision."""

    def test_perfect(self):
        bbs = [BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world")]
        self.assertAlmostEqual(
            word_precision(OCRResult(bounding_boxes=list(bbs)), OCRResult(bounding_boxes=list(bbs))),
            1.0,
        )

    def test_extra_words(self):
        pred = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world foo"),
        ])
        gt = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world"),
        ])
        self.assertAlmostEqual(word_precision(pred, gt), 2 / 3)

    def test_both_empty(self):
        self.assertAlmostEqual(word_precision(OCRResult(), OCRResult()), 1.0)
