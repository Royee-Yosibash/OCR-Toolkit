import math
import unittest

from evaluation.metrics import (
    CharacterAccuracy,
    WordAccuracy,
    word_count_ratio,
    word_precision,
    word_recall,
)
from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr_result import OCRResult


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
    return OCRResult(
        detections=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text=text),
        ]
    )


class TestCharacterAccuracy(unittest.TestCase):
    """Tests for CharacterAccuracy (= exp(-CER))."""

    def setUp(self):
        self.metric = CharacterAccuracy()

    def test_identical(self):
        self.assertAlmostEqual(self.metric(_make_result("hello"), _make_result("hello")), 1.0)

    def test_empty_both(self):
        self.assertAlmostEqual(self.metric(_make_result(""), _make_result("")), 1.0)

    def test_empty_ground_truth(self):
        # 3 spurious characters with empty GT -> CER magnitude = 3 -> exp(-3)
        self.assertAlmostEqual(self.metric(_make_result("abc"), _make_result("")), math.exp(-3.0))

    def test_empty_prediction(self):
        # CER = 3/3 = 1 -> exp(-1)
        self.assertAlmostEqual(self.metric(_make_result(""), _make_result("abc")), math.exp(-1.0))

    def test_substitution(self):
        # CER = 1/3
        self.assertAlmostEqual(self.metric(_make_result("abc"), _make_result("axc")), math.exp(-1 / 3))

    def test_insertion(self):
        # CER = 1/3
        self.assertAlmostEqual(self.metric(_make_result("abcd"), _make_result("abc")), math.exp(-1 / 3))

    def test_is_bounded(self):
        self.assertTrue(self.metric.is_bounded)

    def test_score_in_unit_interval(self):
        # Always in (0, 1] regardless of input.
        score = self.metric(_make_result("xxxxxxxxx"), _make_result("abc"))
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestWordAccuracy(unittest.TestCase):
    """Tests for WordAccuracy (= exp(-WER))."""

    def setUp(self):
        self.metric = WordAccuracy()

    def test_identical(self):
        self.assertAlmostEqual(self.metric(_make_result("hello world"), _make_result("hello world")), 1.0)

    def test_empty_both(self):
        self.assertAlmostEqual(self.metric(_make_result(""), _make_result("")), 1.0)

    def test_empty_ground_truth(self):
        # 2 spurious words with empty GT -> WER magnitude = 2 -> exp(-2)
        self.assertAlmostEqual(self.metric(_make_result("hello world"), _make_result("")), math.exp(-2.0))

    def test_one_substitution(self):
        # WER = 1/2
        self.assertAlmostEqual(
            self.metric(_make_result("hello earth"), _make_result("hello world")),
            math.exp(-0.5),
        )

    def test_completely_wrong(self):
        # WER = 2/2 = 1 -> exp(-1)
        self.assertAlmostEqual(
            self.metric(_make_result("foo bar"), _make_result("hello world")),
            math.exp(-1.0),
        )

    def test_is_bounded(self):
        self.assertTrue(self.metric.is_bounded)

    def test_split_bbs_equivalent(self):
        one_bb = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (20, 10)), text="hello world"),
            ]
        )
        two_bbs = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hello"),
                BoundingBox(coordinates=((10, 0), (20, 10)), text="world"),
            ]
        )
        self.assertAlmostEqual(self.metric(two_bbs, one_bb), 1.0)

    def test_special_chars_ignored(self):
        pred = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hello\nworld"),
            ]
        )
        gt = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world"),
            ]
        )
        self.assertAlmostEqual(self.metric(pred, gt), 1.0)

    def test_punctuation_ignored(self):
        pred = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hello, world."),
            ]
        )
        gt = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world"),
            ]
        )
        self.assertAlmostEqual(self.metric(pred, gt), 1.0)

    def test_ascii_contraction_apostrophe_preserved(self):
        # ASCII apostrophes inside words must not split tokens (don't stays one word).
        pred = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="don't stop"),
            ]
        )
        gt = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="don't stop"),
            ]
        )
        self.assertAlmostEqual(self.metric(pred, gt), 1.0)

    def test_unicode_contraction_apostrophe_preserved(self):
        # Unicode curly apostrophes inside words must not split tokens.
        pred = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="don\u2019t stop"),
            ]
        )
        gt = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="don\u2019t stop"),
            ]
        )
        self.assertAlmostEqual(self.metric(pred, gt), 1.0)

    def test_quoting_apostrophes_stripped(self):
        # Leading/trailing apostrophes (not between word chars) are stripped.
        pred = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="'hello' 'world'"),
            ]
        )
        gt = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world"),
            ]
        )
        self.assertAlmostEqual(self.metric(pred, gt), 1.0)


class TestWordCountRatio(unittest.TestCase):
    """Tests for word_count_ratio."""

    def test_equal_counts(self):
        pred = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world"),
            ]
        )
        gt = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="foo bar"),
            ]
        )
        self.assertAlmostEqual(word_count_ratio(pred, gt), 1.0)

    def test_over_detection(self):
        pred = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="a b c"),
            ]
        )
        gt = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="a b"),
            ]
        )
        self.assertAlmostEqual(word_count_ratio(pred, gt), 1.5)

    def test_both_empty(self):
        self.assertAlmostEqual(word_count_ratio(OCRResult(), OCRResult()), 1.0)


class TestWordRecall(unittest.TestCase):
    """Tests for word_recall."""

    def test_perfect(self):
        dets = [BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world")]
        self.assertAlmostEqual(
            word_recall(OCRResult(detections=list(dets)), OCRResult(detections=list(dets))),
            1.0,
        )

    def test_partial(self):
        pred = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hello"),
            ]
        )
        gt = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world"),
            ]
        )
        self.assertAlmostEqual(word_recall(pred, gt), 0.5)

    def test_split_bbs(self):
        pred = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hello"),
                BoundingBox(coordinates=((10, 0), (20, 10)), text="world"),
            ]
        )
        gt = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (20, 10)), text="hello world"),
            ]
        )
        self.assertAlmostEqual(word_recall(pred, gt), 1.0)

    def test_both_empty(self):
        self.assertAlmostEqual(word_recall(OCRResult(), OCRResult()), 1.0)

    def test_duplicate_words(self):
        pred = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hello"),
            ]
        )
        gt = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hello hello"),
            ]
        )
        self.assertAlmostEqual(word_recall(pred, gt), 0.5)


class TestWordPrecision(unittest.TestCase):
    """Tests for word_precision."""

    def test_perfect(self):
        dets = [BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world")]
        self.assertAlmostEqual(
            word_precision(OCRResult(detections=list(dets)), OCRResult(detections=list(dets))),
            1.0,
        )

    def test_extra_words(self):
        pred = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world foo"),
            ]
        )
        gt = OCRResult(
            detections=[
                BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world"),
            ]
        )
        self.assertAlmostEqual(word_precision(pred, gt), 2 / 3)

    def test_both_empty(self):
        self.assertAlmostEqual(word_precision(OCRResult(), OCRResult()), 1.0)
