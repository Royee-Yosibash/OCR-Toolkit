import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from functools import partial
from pathlib import Path
from typing import Protocol, runtime_checkable

import ocr_backbone.image_preprocessing as image_preprocessing
from ocr_backbone.input_image import InputImage
from ocr_backbone.polygon import Polygon
from utils.json_utils import load_json
from utils.serialize_utils import TYPE_KEY, SerializableClass

PPReturnType = InputImage | list[InputImage]
VALID_PP_RETURN_TYPES = (InputImage, list[InputImage], PPReturnType)


@runtime_checkable
class PreprocessingProtocol(Protocol):
    """Protocol that all preprocessing callables must satisfy.

    A conforming function takes an ``InputImage`` as its first positional
    argument and returns either a single ``InputImage`` or a list of
    ``InputImage`` objects.

    Example::

        def my_step(input_image: InputImage) -> InputImage:
            ...

        def my_splitter(input_image: InputImage) -> list[InputImage]:
            ...
    """

    def __call__(self, input_image: InputImage) -> PPReturnType: ...


@dataclass
class OCRConfig(SerializableClass):
    """Configuration for an OCR run.

    Inherits from ``SerializableClass`` so its instances participate in the
    project-wide registry and can be discriminated by ``_type`` when nested
    in heterogeneous serialized structures. Two of its fields are callables
    that the generic serializer cannot round-trip on its own
    (``detection_validator`` and ``preprocess_methods``); ``to_dict`` and
    ``from_dict`` are overridden to translate them to and from JSON-safe
    descriptors while delegating every other field to the base class.

    Args:
        model_name: Name of the OCR model to use.
        model_params: Model-specific runtime parameters.
        detection_validator: Optional function that validated detection and returns
            True if the detection is valid. Invalid detections are discarded after
            OCR inference.
        preprocess_methods: A list of callables conforming to
            ``PreprocessingProtocol``. When constructed via ``from_dict``,
            method descriptors (dicts with "name" and optional "kwargs")
            are resolved into callables automatically.
    """

    model_name: str
    model_params: dict = field(default_factory=dict)
    detection_validator: Callable[[Polygon], bool] | None = None
    preprocess_methods: list[PreprocessingProtocol] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the config immediately after construction."""
        self.validate()

    def validate(self) -> None:
        """Run all validation checks on the current config state.

        Raises:
            ValueError: If ``model_name`` is empty or ``detection_validator``
                is not callable.
            TypeError: If a preprocessing callable has an incompatible
                signature.
        """
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError(f"model_name must be a non-empty string, got {self.model_name!r}.")

        if self.detection_validator is not None and not callable(self.detection_validator):
            raise TypeError(f"detection_validator must be callable or None, got {type(self.detection_validator)!r}.")

        for method in self.preprocess_methods:
            self._validate_pp_signature(method)

    @staticmethod
    def _resolve_pp_method(pp_method: dict) -> Callable:
        """Resolve a preprocessing method descriptor into a callable.

        If the name contains a dot it is treated as a fully qualified
        dotted path (e.g. ``"my_package.module.func"``).  The last
        segment is the attribute name and everything before it is the
        module path that will be dynamically imported. Otherwise, the
        name is looked up in ``image_preprocessing``.

        Args:
            pp_method: A dict with "name" (function name in
                image_preprocessing, or a dotted module path) and
                optional "kwargs" to bind.

        Returns:
            A callable that accepts an InputImage as its first argument.

        Raises:
            AttributeError: If the function name does not exist in the
                resolved module.
            ModuleNotFoundError: If the dotted module path cannot be
                imported.
        """
        name = pp_method["name"]
        if "." in name:
            module_path, attr_name = name.rsplit(".", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, attr_name)
        else:
            func = getattr(image_preprocessing, name)
        kwargs = pp_method.get("kwargs", {})
        return partial(func, **kwargs) if kwargs else func

    @staticmethod
    def _validate_pp_signature(method: Callable) -> None:
        """Validate that a preprocessing callable has a compatible signature.

        The first unbound parameter must be annotated as ``InputImage``
        and the return must be annotated as ``InputImage``,
        ``list[InputImage]``, or ``InputImage | list[InputImage]``.

        Args:
            method: The preprocessing callable to validate.

        Raises:
            TypeError: If the signature is incompatible or missing
                required annotations.
        """
        sig = inspect.signature(method)
        params = list(sig.parameters.values())

        if not params:
            raise TypeError(
                f"Preprocessing method {method!r} accepts no arguments; expected at least one (InputImage)."
            )

        first_param = params[0]
        ann = first_param.annotation
        if ann is inspect.Parameter.empty:
            raise TypeError(f"Preprocessing method {method!r}: first parameter must be annotated as InputImage.")
        if ann is not InputImage:
            raise TypeError(
                f"Preprocessing method {method!r}: first parameter is annotated as {ann!r}, expected InputImage."
            )

        ret = sig.return_annotation
        if ret is inspect.Signature.empty:
            raise TypeError(
                f"Preprocessing method {method!r}: missing return annotation, expected InputImage or list[InputImage]."
            )
        if ret not in VALID_PP_RETURN_TYPES:
            raise TypeError(
                f"Preprocessing method {method!r}: return annotation "
                f"is {ret!r}, expected InputImage or list[InputImage]."
            )

    def update(self, overrides: dict) -> None:
        """Update config attributes from a dict.

        Only keys that correspond to existing dataclass fields are applied.
        Unknown keys are ignored. If ``preprocess_methods`` is provided as
        a list of dicts, each entry is resolved into a callable.

        Args:
            overrides: A dict mapping field names to new values.
        """
        valid_names = {f.name for f in fields(self)}
        for key, value in overrides.items():
            if key in valid_names:
                if key == "preprocess_methods":
                    value = [self._resolve_pp_method(m) if isinstance(m, dict) else m for m in value]
                setattr(self, key, value)
            else:
                self.model_params[key] = value

        self.validate()

    def to_dict(self) -> dict:
        """Convert the config to a JSON-serializable dict.

        Delegates to ``SerializableClass.to_dict`` for the trivial fields,
        then rewrites the two callable-valued fields into JSON-safe
        descriptors. ``detection_validator`` is stored as a
        ``"module.path:function_name"`` string when present and omitted
        when ``None``. Each preprocessing callable is serialized to a
        dict with ``"name"`` as a fully qualified dotted path and
        optional ``"kwargs"``.

        Returns:
            A plain dict representation of this config.
        """
        data = super().to_dict()
        data["preprocess_methods"] = [self._serialize_pp_method(m) for m in self.preprocess_methods]
        if self.detection_validator is None:
            data.pop("detection_validator", None)
        else:
            module = self.detection_validator.__module__
            qualname = self.detection_validator.__qualname__
            data["detection_validator"] = f"{module}:{qualname}"
        return data

    @staticmethod
    def _serialize_pp_method(method: Callable) -> dict:
        """Serialize a preprocessing callable into a descriptor dict.

        Args:
            method: The preprocessing callable, optionally wrapped in
                ``functools.partial`` to carry bound kwargs.

        Returns:
            A dict with ``"name"`` (the fully qualified dotted path of
            the underlying function) and an optional ``"kwargs"`` mapping.
        """
        func = method.func if isinstance(method, partial) else method
        kwargs = method.keywords if isinstance(method, partial) else {}
        entry: dict = {"name": f"{func.__module__}.{func.__qualname__}"}
        if kwargs:
            entry["kwargs"] = kwargs
        return entry

    @classmethod
    def from_dict(cls, raw_dict: dict):
        """Create an OCRConfig from a plain dict.

        Resolves the two callable-valued fields into live callables and
        then defers to ``SerializableClass.from_dict`` for instantiation.
        ``detection_validator`` may be a callable or a dotted-path string
        in the form ``"module.path:function_name"``. ``preprocess_methods``
        entries that are dicts are resolved via ``_resolve_pp_method``.
        Keys that do not correspond to a dataclass field are silently
        dropped (besides ``_type``), preserving backward compatibility
        with config JSONs that carry extra metadata such as ``alias``.

        Args:
            raw_dict: Dict with at least ``model_name`` and optionally
                ``model_params``, ``detection_validator``, and
                ``preprocess_methods``.

        Returns:
            An OCRConfig instance.

        Raises:
            TypeError: If ``model_name`` is missing from ``raw_dict``.
        """
        valid_names = {f.name for f in fields(cls)}
        prepared = {k: v for k, v in raw_dict.items() if k == TYPE_KEY or k in valid_names}

        detection_validator = prepared.get("detection_validator")
        if isinstance(detection_validator, str):
            module_path, attr_name = detection_validator.rsplit(":", 1)
            module = importlib.import_module(module_path)
            prepared["detection_validator"] = getattr(module, attr_name)

        raw_pp = prepared.get("preprocess_methods")
        if raw_pp is not None:
            prepared["preprocess_methods"] = [cls._resolve_pp_method(m) if isinstance(m, dict) else m for m in raw_pp]

        return super().from_dict(prepared)


def load_config(path: str | Path) -> OCRConfig:
    """Load an OCR configuration from a JSON file.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        An OCRConfig instance populated from the file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If required fields are missing from the JSON.
    """
    return OCRConfig.from_dict(load_json(path))
