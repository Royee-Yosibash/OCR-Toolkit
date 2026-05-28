"""Helpers for serializing, resolving, and validating callables as JSON-safe descriptors.

A *callable descriptor* is a plain dict of the form::

    {"name": "<dotted.path.or.bare_name>", "kwargs": {...}}

where ``kwargs`` is optional. These helpers convert between such
descriptors and live callables, and validate that a callable's
signature matches a required first-parameter type and return type.
"""

import importlib
import inspect
from collections.abc import Callable
from functools import partial
from types import ModuleType


def resolve_callable_descriptor(descriptor: dict, *, default_module: ModuleType | None = None) -> Callable:
    """Resolve a callable descriptor dict into a live callable.

    If the name contains a dot it is treated as a fully qualified dotted
    path (e.g. ``"my_package.module.func"``). The last segment is the
    attribute name and everything before it is the module path that will
    be dynamically imported. Otherwise, the name is looked up in
    ``default_module``.

    Args:
        descriptor: A dict with "name" (a dotted module path or a bare
            attribute name resolvable in ``default_module``) and
            optional "kwargs" to bind.
        default_module: Module used to resolve bare names. If ``None``,
            a dotted path is required.

    Returns:
        A callable, optionally wrapped in ``functools.partial`` when
        ``kwargs`` are present.

    Raises:
        ValueError: If ``name`` is a bare attribute and ``default_module``
            is ``None``.
        AttributeError: If the function name does not exist in the
            resolved module.
        ModuleNotFoundError: If the dotted module path cannot be imported.
    """
    name = descriptor["name"]
    if "." in name:
        module_path, attr_name = name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        func = getattr(module, attr_name)
    elif default_module is not None:
        func = getattr(default_module, name)
    else:
        raise ValueError(
            f"Callable descriptor name '{name}' must be a fully qualified dotted path "
            f"when no default module is provided."
        )
    kwargs = descriptor.get("kwargs", {})
    return partial(func, **kwargs) if kwargs else func


def serialize_callable_descriptor(method: Callable) -> dict:
    """Serialize a callable into a descriptor dict.

    Args:
        method: The callable, optionally wrapped in ``functools.partial``
            to carry bound kwargs.

    Returns:
        A dict with ``"name"`` (the fully qualified dotted path of the
        underlying function) and an optional ``"kwargs"`` mapping.
    """
    func = method.func if isinstance(method, partial) else method
    kwargs = method.keywords if isinstance(method, partial) else {}
    entry: dict = {"name": f"{func.__module__}.{func.__qualname__}"}
    if kwargs:
        entry["kwargs"] = kwargs
    return entry


def validate_callable_signature(
    method: Callable,
    *,
    expected_first_param: type,
    valid_returns: tuple,
) -> None:
    """Validate that a callable has a compatible signature.

    The first unbound parameter must be annotated as
    ``expected_first_param`` and the return annotation must be one of
    ``valid_returns``.

    Args:
        method: The callable to validate.
        expected_first_param: The required annotation type for the
            callable's first parameter.
        valid_returns: A tuple of acceptable return annotations.

    Raises:
        TypeError: If the signature is incompatible or missing required
            annotations.
    """
    sig = inspect.signature(method)
    params = list(sig.parameters.values())

    if not params:
        raise TypeError(f"{method!r} accepts no arguments; expected at least one ({expected_first_param.__name__}).")

    ann = params[0].annotation
    if ann is inspect.Parameter.empty:
        raise TypeError(f"{method!r}: first parameter must be annotated as {expected_first_param.__name__}.")
    if ann is not expected_first_param:
        raise TypeError(
            f"{method!r}: first parameter is annotated as {ann!r}, expected {expected_first_param.__name__}."
        )

    ret = sig.return_annotation
    if ret is inspect.Signature.empty:
        raise TypeError(f"{method!r}: missing return annotation, expected one of {valid_returns!r}.")
    if ret not in valid_returns:
        raise TypeError(f"{method!r}: return annotation is {ret!r}, expected one of {valid_returns!r}.")
