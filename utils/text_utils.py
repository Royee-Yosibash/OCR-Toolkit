"""Text normalization helpers for OCR output processing."""

import re

_PUNCTUATION_RE = re.compile(
    r"(?!(?<=\w)['\u2018\u2019](?=\w))(?:(?<!\d)[^\w\s]|[^\w\s](?!\d))",
    re.UNICODE,
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for comparison-based metrics.

    Strips punctuation while preserving intra-word apostrophes
    (e.g. "don't"), collapses all whitespace (including newlines and
    tabs) into single spaces, and strips leading/trailing whitespace.

    Args:
        text: Raw text string.

    Returns:
        Normalized text string.
    """
    text = _PUNCTUATION_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
