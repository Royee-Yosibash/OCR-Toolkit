"""Core OCR abstractions package.

The re-exports below expose the package's public types and, as a
necessary side effect, ensure every ``SerializableClass`` subclass in
this package registers itself in ``SerializableClass._registry`` (via
``__init_subclass__``) the moment any consumer touches the package.
Add a line here when introducing a new ``SerializableClass`` subclass
under ``ocr_backbone/``.
"""

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_result import OCRResult
from ocr_backbone.polygon import Polygon

__all__ = ["BoundingBox", "OCRConfig", "OCRResult", "Polygon"]
