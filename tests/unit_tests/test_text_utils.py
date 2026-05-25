import unittest

from utils.text_utils import normalize_text


class TestNormalizeText(unittest.TestCase):
    def test_strips_punctuation(self):
        self.assertEqual(normalize_text("hello, world!"), "hello world")

    def test_preserves_intra_word_apostrophe(self):
        self.assertEqual(normalize_text("don't stop"), "don't stop")

    def test_preserves_unicode_apostrophe(self):
        self.assertEqual(normalize_text("don\u2019t stop"), "don\u2019t stop")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize_text("a\t\nb   c"), "a b c")

    def test_strips_leading_and_trailing_whitespace(self):
        self.assertEqual(normalize_text("  hello  "), "hello")

    def test_preserves_intra_number_separators(self):
        self.assertEqual(normalize_text("3.14 and 1,000"), "3.14 and 1,000")

    def test_empty_input(self):
        self.assertEqual(normalize_text(""), "")
