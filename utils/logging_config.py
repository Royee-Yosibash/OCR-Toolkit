"""Shared logging configuration for OCR-Toolkit entry points."""

import logging

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Apply the standard root-logger configuration used by all entry points.

    Calls ``logging.basicConfig`` with the toolkit's shared format and
    ``force=True`` so the configuration overrides any previously installed
    handlers (e.g. those installed by third-party libraries at import time).

    Args:
        level: The root logger level to install. Defaults to ``logging.INFO``.

    Returns:
        None.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
