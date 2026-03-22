"""OCR module implementations package.

Provides concrete OCR engine wrappers that register themselves as
subclasses of ``OCRAbstract`` on import.
"""

import importlib
import pkgutil
from pathlib import Path


def import_all_modules() -> None:
    """Import every module inside this package to trigger subclass
    registration in ``OCRAbstract._registry``.
    """
    package_path = Path(__file__).parent
    for _, name, _ in pkgutil.iter_modules([str(package_path)]):
        importlib.import_module(f"ocr_modules.{name}")
