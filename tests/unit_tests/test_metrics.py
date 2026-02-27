import unittest

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr_result import OCRResult
from evaluation.metrics import (
    character_error_rate,
    word_error_rate,
    ocr_result_cer,
    ocr_result_wer,
    word_count_ratio,
    word_recall,
    word_precision,
)


class TestCharacterErrorRate(unittest.TestCase):
    """Tests for character_error_rate."""

    def test_identical(self):
        self.assertAlmostEqual(character_error_rate("hello", "hello"), 0.0)

    def test_empty_both(self):
        self.assertAlmostEqual(character_error_rate("", ""), 0.0)

    def test_empty_ground_truth(self):
        self.assertEqual(character_error_rate("abc", ""), 3.0)

    def test_empty_prediction(self):
        self.assertAlmostEqual(character_error_rate("", "abc"), 1.0)

    def test_substitution(self):
        self.assertAlmostEqual(character_error_rate("abc", "axc"), 1 / 3)

    def test_insertion(self):
        self.assertAlmostEqual(character_error_rate("abcd", "abc"), 1 / 3)


class TestWordErrorRate(unittest.TestCase):
    """Tests for word_error_rate."""

    def test_identical(self):
        self.assertAlmostEqual(word_error_rate("hello world", "hello world"), 0.0)

    def test_empty_both(self):
        self.assertAlmostEqual(word_error_rate("", ""), 0.0)

    def test_empty_ground_truth(self):
        self.assertEqual(word_error_rate("hello world", ""), 2.0)

    def test_one_substitution(self):
        self.assertAlmostEqual(word_error_rate("hello earth", "hello world"), 0.5)

    def test_completely_wrong(self):
        self.assertAlmostEqual(word_error_rate("foo bar", "hello world"), 1.0)


class TestOCRResultCER(unittest.TestCase):
    """Tests for ocr_result_cer."""

    def test_identical_results(self):
        bbs = [BoundingBox(coordinates=((0, 0), (10, 10)), text="hello")]
        self.assertAlmostEqual(
            ocr_result_cer(OCRResult(bounding_boxes=list(bbs)), OCRResult(bounding_boxes=list(bbs))),
            0.0,
        )


class TestOCRResultWER(unittest.TestCase):
    """Tests for ocr_result_wer."""

    def test_identical_results(self):
        bbs = [BoundingBox(coordinates=((0, 0), (10, 10)), text="hello")]
        self.assertAlmostEqual(
            ocr_result_wer(OCRResult(bounding_boxes=list(bbs)), OCRResult(bounding_boxes=list(bbs))),
            0.0,
        )

    def test_split_bbs_equivalent(self):
        one_bb = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (20, 10)), text="hello world"),
        ])
        two_bbs = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello"),
            BoundingBox(coordinates=((10, 0), (20, 10)), text="world"),
        ])
        self.assertAlmostEqual(ocr_result_wer(two_bbs, one_bb), 0.0)

    def test_special_chars_ignored(self):
        pred = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello\nworld"),
        ])
        gt = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world"),
        ])
        self.assertAlmostEqual(ocr_result_wer(pred, gt), 0.0)

    def test_punctuation_ignored(self):
        pred = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello, world."),
        ])
        gt = OCRResult(bounding_boxes=[
            BoundingBox(coordinates=((0, 0), (10, 10)), text="hello world"),
        ])
        self.assertAlmostEqual(ocr_result_wer(pred, gt), 0.0)


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
