"""Evaluation pipeline, metrics, and ground-truth types.

The re-export below exposes the package's public ground-truth type
and, as a necessary side effect, ensures every ``SerializableClass``
subclass in this package registers itself in
``SerializableClass._registry`` (via ``__init_subclass__``) the moment
any consumer touches the package. Add a line here when introducing a
new ``SerializableClass`` subclass under ``evaluation/``.
"""

from evaluation.ocr_ground_truth import OCRGroundTruth

__all__ = ["OCRGroundTruth"]
