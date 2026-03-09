"""Lazy module import utility.

Provides a proxy object that defers the actual ``importlib.import_module``
call until the first attribute access, avoiding upfront import cost for
heavy dependencies.
"""

import importlib


class LazyModule:
    """Proxy that lazily imports a module on first attribute access.

    Args:
        module_path: Fully qualified module name (e.g. ``"scipy.stats"``).
    """

    def __init__(self, module_path: str) -> None:
        object.__setattr__(self, "_module_path", module_path)
        object.__setattr__(self, "_module", None)

    def __getattr__(self, name: str):
        """Import the module on first access and delegate attribute lookup.

        Args:
            name: Attribute name to retrieve from the lazily loaded module.

        Returns:
            The requested attribute from the underlying module.
        """
        module = object.__getattribute__(self, "_module")
        if module is None:
            module_path = object.__getattribute__(self, "_module_path")
            module = importlib.import_module(module_path)
            object.__setattr__(self, "_module", module)
        return getattr(module, name)
