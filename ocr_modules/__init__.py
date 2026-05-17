"""OCR module implementations package.

Provides concrete OCR engine wrappers that register themselves as
subclasses of ``OCRAbstract`` on import.

Conditional registration contract
---------------------------------
Each module in this package is REQUIRED to gate its concrete
``OCRAbstract`` subclass behind an ``importlib.util.find_spec("<engine>")``
availability check followed by ``if _HAS_<ENGINE>: class ...:``. This
keeps subclasses whose engine is not installed out of
``OCRAbstract._registry`` and out of ``OCRAbstract.registered_models()``.

The actual heavy engine import is deferred via ``utils.lazy_import.LazyModule``
so that calling :func:`import_all_modules` only pays the cost of populating
the registry, not of importing every engine's runtime (PaddlePaddle, Torch,
Tesseract bindings). The engine import fires lazily on first attribute
access -- typically inside the subclass's ``__init__``.

Downstream consumers (e.g. the tagging UI's model dropdown and the
default-mode of ``run_evaluation``) rely on this gating so that
unavailable engines are never offered to the user. Do not collapse the
pattern by always defining the class without the ``_HAS_<ENGINE>`` guard.
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
