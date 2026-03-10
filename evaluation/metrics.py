"""OCR evaluation metrics for comparing predicted results against ground truth.

Provides text-level metrics (CER, WER) and word-level bag-of-words
metrics for evaluating OCR accuracy independent of bounding box geometry.
"""

from collections.abc import Callable, Sequence
import re
from collections import Counter

from ocr_backbone.ocr_result import OCRResult

_PUNCTUATION_RE = re.compile(r"(?<!\d)[^\w\s]|[^\w\s](?!\d)", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")

MetricFn = Callable[[OCRResult, OCRResult], float | int | bool]


def bounded_metric(fn: MetricFn) -> MetricFn:
    """Mark a metric function as bounded to [0, 1].

    Args:
        fn: The metric function to decorate.

    Returns:
        The same function with ``is_bounded`` set to True.
    """
    fn.is_bounded = True
    return fn


def unbounded_metric(fn: MetricFn) -> MetricFn:
    """Mark a metric function as unbounded (can exceed [0, 1]).

    Args:
        fn: The metric function to decorate.

    Returns:
        The same function with ``is_bounded`` set to False.
    """
    fn.is_bounded = False
    return fn


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


def character_error_rate(prediction: str, ground_truth: str) -> float:
    """Compute the Character Error Rate (CER).

    CER = levenshtein_distance(prediction, ground_truth) / len(ground_truth)

    Args:
        prediction: The predicted text string.
        ground_truth: The reference text string.

    Returns:
        The CER as a float in [0, inf). Returns 0.0 when both strings
        are empty.
    """
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else float(len(prediction))
    return _levenshtein_distance(prediction, ground_truth) / len(ground_truth)


def word_error_rate(prediction: str, ground_truth: str) -> float:
    """Compute the Word Error Rate (WER).

    Splits both strings on whitespace and computes the Levenshtein
    distance at the word level, normalized by the number of words in
    the ground truth.

    Args:
        prediction: The predicted text string.
        ground_truth: The reference text string.

    Returns:
        The WER as a float in [0, inf). Returns 0.0 when both strings
        are empty.
    """
    gt_words = ground_truth.split()
    pred_words = prediction.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else float(len(pred_words))
    return _levenshtein_distance(pred_words, gt_words) / len(gt_words)


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
    """Concatenate and normalize all BB texts from an OCRResult.

    Args:
        result: An OCRResult with sorted bounding boxes.

    Returns:
        Normalized space-separated string of all BB texts.
    """
    raw = " ".join(bb.text for bb in result.bounding_boxes)
    return _normalize_text(raw)


def _ocr_result_to_words(result: OCRResult) -> list[str]:
    """Extract all individual words from an OCRResult after normalization.

    Args:
        result: An OCRResult.

    Returns:
        A flat list of all words across all bounding boxes.
    """
    return _ocr_result_to_text(result).split()


@unbounded_metric
def ocr_result_cer(prediction: OCRResult, ground_truth: OCRResult) -> float:
    """Compute the CER over full concatenated text of two OCRResults.

    Args:
        prediction: The predicted OCR result.
        ground_truth: The reference OCR result.

    Returns:
        The CER as a float.
    """
    return character_error_rate(
        _ocr_result_to_text(prediction),
        _ocr_result_to_text(ground_truth),
    )


@unbounded_metric
def ocr_result_wer(prediction: OCRResult, ground_truth: OCRResult) -> float:
    """Compute the WER over full concatenated text of two OCRResults.

    Args:
        prediction: The predicted OCR result.
        ground_truth: The reference OCR result.

    Returns:
        The WER as a float.
    """
    return word_error_rate(
        _ocr_result_to_text(prediction),
        _ocr_result_to_text(ground_truth),
    )


@unbounded_metric
def word_count_ratio(prediction: OCRResult, ground_truth: OCRResult) -> float:
    """Compute the ratio of predicted word count to ground truth word count.

    A value of 1.0 means the same number of words were detected.
    Values above 1.0 indicate over-detection, below 1.0 indicate
    missed words.

    Args:
        prediction: The predicted OCR result.
        ground_truth: The reference OCR result.

    Returns:
        The ratio as a float. Returns 1.0 when both are empty, 0.0
        when prediction is empty but GT is not.
    """
    gt_words = _ocr_result_to_words(ground_truth)
    pred_words = _ocr_result_to_words(prediction)
    if len(gt_words) == 0:
        return 1.0 if len(pred_words) == 0 else float(len(pred_words))
    return len(pred_words) / len(gt_words)


@bounded_metric
def word_recall(prediction: OCRResult, ground_truth: OCRResult) -> float:
    """Compute the fraction of ground truth words found in the prediction.

    Uses bag-of-words matching with multiplicity: each GT word is
    matched at most as many times as it appears in the prediction.
    Order and BB boundaries are ignored.

    Args:
        prediction: The predicted OCR result.
        ground_truth: The reference OCR result.

    Returns:
        Recall as a float in [0, 1]. Returns 1.0 when both are empty.
    """
    gt_words = _ocr_result_to_words(ground_truth)
    if len(gt_words) == 0:
        return 1.0
    pred_counts = Counter(_ocr_result_to_words(prediction))
    gt_counts = Counter(gt_words)
    matched = sum(min(pred_counts[w], gt_counts[w]) for w in gt_counts)
    return matched / len(gt_words)


@bounded_metric
def word_precision(prediction: OCRResult, ground_truth: OCRResult) -> float:
    """Compute the fraction of predicted words found in the ground truth.

    Uses bag-of-words matching with multiplicity: each predicted word
    is matched at most as many times as it appears in the GT.
    Order and BB boundaries are ignored.

    Args:
        prediction: The predicted OCR result.
        ground_truth: The reference OCR result.

    Returns:
        Precision as a float in [0, 1]. Returns 1.0 when both are empty.
    """
    pred_words = _ocr_result_to_words(prediction)
    if len(pred_words) == 0:
        return 1.0
    gt_counts = Counter(_ocr_result_to_words(ground_truth))
    pred_counts = Counter(pred_words)
    matched = sum(min(pred_counts[w], gt_counts[w]) for w in pred_counts)
    return matched / len(pred_words)
