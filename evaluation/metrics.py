"""OCR evaluation metrics for comparing predicted results against ground truth.

Provides text-level metrics (CER, WER) and word-level bag-of-words
metrics for evaluating OCR accuracy independent of bounding box geometry.
"""

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence

from evaluation.ocr_ground_truth import OCRGroundTruth
from ocr_backbone.ocr_result import OCRResult

_PUNCTUATION_RE = re.compile(r"(?<!\d)[^\w\s]|[^\w\s](?!\d)", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")

METRICS_BOUNDED_LOOKUP: dict[str, bool] = {}
METRICS_DISPLAY_NAME_LOOKUP: dict[str, str] = {}


class Metric(ABC):
    """Base class for OCR evaluation metrics.

    Subclasses must set ``is_bounded`` and ``display_name``, and implement
    ``__call__``.  Concrete subclasses are automatically registered in
    ``METRICS_BOUNDED_LOOKUP`` and ``METRICS_DISPLAY_NAME_LOOKUP`` via
    ``__init_subclass__``.

    Attributes:
        is_bounded: True if the metric value is confined to [0, 1],
            False if it can exceed that range.
        display_name: Human-readable label used in plots and reports.
    """

    is_bounded: bool
    display_name: str

    def __init_subclass__(cls, **kwargs):
        """Register concrete subclasses in METRICS_BOUNDED_LOOKUP and METRICS_DISPLAY_NAME_LOOKUP."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "is_bounded"):
            METRICS_BOUNDED_LOOKUP[cls.__name__] = cls.is_bounded
        if hasattr(cls, "display_name"):
            METRICS_DISPLAY_NAME_LOOKUP[cls.__name__] = cls.display_name

    @abstractmethod
    def __call__(self, prediction: OCRResult, ground_truth: OCRGroundTruth) -> float:
        """Compute the metric for a prediction/ground-truth pair.

        Args:
            prediction: The predicted OCR result.
            ground_truth: The reference ground truth.

        Returns:
            The metric value as a float.
        """


def _levenshtein_distance(s1: Sequence, s2: Sequence) -> int:
    """Compute the Levenshtein (edit) distance between two sequences.

    Works on any indexable sequences (strings, lists of words, etc.).

    Args:
        s1: First sequence.
        s2: Second sequence.

    Returns:
        The minimum number of single-element edits (insertions,
        deletions, substitutions) needed to transform s1 into s2.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev_row[j + 1] + 1
            delete = curr_row[j] + 1
            substitute = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insert, delete, substitute))
        prev_row = curr_row

    return prev_row[-1]


def _normalize_text(text: str) -> str:
    """Normalize text for metric comparison.

    Strips punctuation, collapses all whitespace (including special
    characters like newlines and tabs) into single spaces, and strips
    leading/trailing whitespace.

    Args:
        text: Raw text string.

    Returns:
        Normalized text string.
    """
    text = _PUNCTUATION_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _ocr_result_to_text(result: OCRResult) -> str:
    """Concatenate and normalize all detection texts from an OCRResult.

    Args:
        result: An OCRResult with sorted bounding boxes.

    Returns:
        Normalized space-separated string of all detection texts.
    """
    raw = " ".join(det.text for det in result.detections)
    return _normalize_text(raw)


def _ocr_result_to_words(result: OCRResult) -> list[str]:
    """Extract all individual words from an OCRResult after normalization.

    Args:
        result: An OCRResult.

    Returns:
        A flat list of all words across all bounding boxes.
    """
    return _ocr_result_to_text(result).split()


class CharacterAccuracy(Metric):
    """Compute a character-level accuracy score from concatenated OCR texts.

    Internally derives the Character Error Rate
    ``CER = levenshtein_distance(pred, gt) / len(gt)``, then maps it to
    an accuracy score via ``exp(-CER)``. When the ground truth is empty
    the raw character count of the prediction is used as the error
    magnitude so spurious detections still produce vanishing scores.

    Best score: 1.0 -- the prediction matches the ground truth character
    for character (CER = 0). The score decays smoothly toward 0 as the
    error rate grows: CER = 0.5 -> ~0.61, CER = 1 -> ~0.37, CER = 2 ->
    ~0.14, CER -> infinity -> 0. The score is bounded to (0, 1] and is
    never exactly 0, preserving differences between very-bad and
    catastrophically-bad predictions.
    """

    is_bounded = True
    display_name = "Character Accuracy"

    def __call__(self, prediction: OCRResult, ground_truth: OCRGroundTruth) -> float:
        pred_text = _ocr_result_to_text(prediction)
        gt_text = _ocr_result_to_text(ground_truth)
        if len(gt_text) == 0:
            cer = 0.0 if len(pred_text) == 0 else float(len(pred_text))
        else:
            cer = _levenshtein_distance(pred_text, gt_text) / len(gt_text)
        return math.exp(-cer)


class WordAccuracy(Metric):
    """Compute a word-level accuracy score from concatenated OCR texts.

    Splits both texts on whitespace and computes the Levenshtein distance
    at the word level, normalized by the number of ground-truth words to
    yield the Word Error Rate (WER). The accuracy score is then
    ``exp(-WER)``. When the ground truth is empty the raw predicted word
    count is used as the error magnitude.

    Best score: 1.0 -- every ground-truth word appears in the same order
    and form in the prediction (WER = 0). The score decays smoothly
    toward 0 as the word-level error grows: WER = 0.5 -> ~0.61,
    WER = 1 -> ~0.37, WER -> infinity -> 0. Bounded to (0, 1]; never
    exactly 0, so it still distinguishes between bad and worse
    predictions.
    """

    is_bounded = True
    display_name = "Word Accuracy"

    def __call__(self, prediction: OCRResult, ground_truth: OCRGroundTruth) -> float:
        pred_words = _ocr_result_to_text(prediction).split()
        gt_words = _ocr_result_to_text(ground_truth).split()
        if len(gt_words) == 0:
            wer = 0.0 if len(pred_words) == 0 else float(len(pred_words))
        else:
            wer = _levenshtein_distance(pred_words, gt_words) / len(gt_words)
        return math.exp(-wer)


class WordCountRatio(Metric):
    """Compute the ratio of predicted word count to ground truth word count.

    A value of 1.0 means the same number of words were detected.
    Values above 1.0 indicate over-detection, below 1.0 indicate
    missed words.

    Best score: 1.0 -- the prediction contains exactly as many words as
    the ground truth (note this only checks counts, not which words).
    Values >1.0 indicate over-detection (hallucinated, duplicated, or
    over-segmented words); values <1.0 indicate under-detection (missed
    or merged words). 0.0 means no words were detected. The metric is
    unbounded above and is best read alongside WordPrecision/WordRecall:
    a ratio of 1.0 with low precision/recall means the right *amount*
    of words but the wrong ones.

    Intentionally left unbounded and two-sided: the sign of the
    deviation from 1.0 is itself the diagnostic signal.
    """

    is_bounded = False
    display_name = "Word Count Ratio"

    def __call__(self, prediction: OCRResult, ground_truth: OCRGroundTruth) -> float:
        gt_words = _ocr_result_to_words(ground_truth)
        pred_words = _ocr_result_to_words(prediction)
        if len(gt_words) == 0:
            return 1.0 if len(pred_words) == 0 else float(len(pred_words))
        return len(pred_words) / len(gt_words)


class WordRecall(Metric):
    """Compute the fraction of ground truth words found in the prediction.

    Uses bag-of-words matching with multiplicity: each GT word is
    matched at most as many times as it appears in the prediction.
    Order and BB boundaries are ignored.

    Best score: 1.0 -- every ground-truth word (with multiplicity) is
    present in the prediction; the OCR missed nothing. Lower values
    indicate progressively more GT words went undetected or were
    transcribed incorrectly: 0.5 means half the GT words were
    recovered, 0.0 means none were. Bounded to [0, 1]. Recall is
    insensitive to spurious extra words in the prediction; pair it
    with WordPrecision to detect over-detection.
    """

    is_bounded = True
    display_name = "Word Recall"

    def __call__(self, prediction: OCRResult, ground_truth: OCRGroundTruth) -> float:
        gt_words = _ocr_result_to_words(ground_truth)
        if len(gt_words) == 0:
            return 1.0
        pred_counts = Counter(_ocr_result_to_words(prediction))
        gt_counts = Counter(gt_words)
        matched = sum(min(pred_counts[w], gt_counts[w]) for w in gt_counts)
        return matched / len(gt_words)


class WordPrecision(Metric):
    """Compute the fraction of predicted words found in the ground truth.

    Uses bag-of-words matching with multiplicity: each predicted word
    is matched at most as many times as it appears in the GT.
    Order and BB boundaries are ignored.

    Best score: 1.0 -- every predicted word (with multiplicity) is also
    in the ground truth; the OCR added nothing spurious. Lower values
    mean a growing fraction of predicted words are hallucinated,
    duplicated, or mis-recognized: 0.5 means half the predicted words
    do not appear in the GT, 0.0 means none of the prediction matches.
    Bounded to [0, 1]. Precision is insensitive to GT words the model
    failed to find; pair it with WordRecall to detect under-detection.
    """

    is_bounded = True
    display_name = "Word Precision"

    def __call__(self, prediction: OCRResult, ground_truth: OCRGroundTruth) -> float:
        pred_words = _ocr_result_to_words(prediction)
        if len(pred_words) == 0:
            return 1.0
        gt_counts = Counter(_ocr_result_to_words(ground_truth))
        pred_counts = Counter(pred_words)
        matched = sum(min(pred_counts[w], gt_counts[w]) for w in pred_counts)
        return matched / len(pred_words)


def metric_class_display_name(name: str) -> str:
    """Convert a metric class name to a human-readable label.

    Uses the ``display_name`` registered by each Metric subclass when
    available, otherwise falls back to title-casing the class name.

    Args:
        name: The metric class name (e.g. ``CharacterAccuracy``).

    Returns:
        Human-readable label (e.g. ``Character Accuracy``).
    """
    return METRICS_DISPLAY_NAME_LOOKUP.get(name, name)


ocr_result_char_accuracy = CharacterAccuracy()
ocr_result_word_accuracy = WordAccuracy()
word_count_ratio = WordCountRatio()
word_recall = WordRecall()
word_precision = WordPrecision()
