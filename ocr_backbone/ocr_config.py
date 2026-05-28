from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Protocol, runtime_checkable

import ocr_backbone.image_preprocessing as image_preprocessing
from ocr_backbone.input_image import InputImage
from ocr_backbone.polygon import Polygon
from utils.callable_descriptors import (
    resolve_callable_descriptor,
    serialize_callable_descriptor,
    validate_callable_signature,
)
from utils.json_utils import load_json
from utils.serialize_utils import TYPE_KEY, SerializableClass

PPReturnType = InputImage | list[InputImage]
VALID_PP_RETURN_TYPES = (InputImage, list[InputImage], PPReturnType)
VALID_VALIDATOR_RETURN_TYPES = (bool,)


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


@runtime_checkable
class DetectionValidatorProtocol(Protocol):
    """Protocol that all detection validators must satisfy.

    A conforming function takes a ``Polygon`` as its first positional
    argument and returns ``True`` if the detection should be kept,
    ``False`` if it should be filtered out. A detection passes overall
    validation only when every validator in the list returns ``True``.

    Example::

        def keep_high_confidence(detection: Polygon) -> bool:
            return detection.confidence >= 0.5
    """

    def __call__(self, detection: Polygon) -> bool: ...


@dataclass
class OCRConfig(SerializableClass):
    """Configuration for an OCR run.

    Inherits from ``SerializableClass`` so its instances participate in the
    project-wide registry and can be discriminated by ``_type`` when nested
    in heterogeneous serialized structures. Two of its fields are lists of
    callables that the generic serializer cannot round-trip on its own
    (``detection_validators`` and ``preprocess_methods``); ``to_dict`` and
    ``from_dict`` are overridden to translate them to and from JSON-safe
    descriptors while delegating every other field to the base class.

    Args:
        model_name: Name of the OCR model to use.
        model_params: Model-specific runtime parameters.
        detection_validators: A list of callables conforming to
            ``DetectionValidatorProtocol``. A detection is kept only when
            every validator returns ``True``; an empty list accepts all
            detections.
        preprocess_methods: A list of callables conforming to
            ``PreprocessingProtocol``. When constructed via ``from_dict``,
            method descriptors (dicts with "name" and optional "kwargs")
            are resolved into callables automatically.
    """

    model_name: str
    model_params: dict = field(default_factory=dict)
    detection_validators: list[DetectionValidatorProtocol] = field(default_factory=list)
    preprocess_methods: list[PreprocessingProtocol] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the config immediately after construction."""
        self.validate()

    def validate(self) -> None:
        """Run all validation checks on the current config state.

        Raises:
            ValueError: If ``model_name`` is empty.
            TypeError: If a preprocessing callable or detection validator
                has an incompatible signature.
        """
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError(f"model_name must be a non-empty string, got {self.model_name!r}.")

        for method in self.preprocess_methods:
            validate_callable_signature(
                method,
                expected_first_param=InputImage,
                valid_returns=VALID_PP_RETURN_TYPES,
            )
        for validator in self.detection_validators:
            validate_callable_signature(
                validator,
                expected_first_param=Polygon,
                valid_returns=VALID_VALIDATOR_RETURN_TYPES,
            )

    def update(self, overrides: dict) -> None:
        """Update config attributes from a dict.

        Only keys that correspond to existing dataclass fields are applied.
        Unknown keys are merged into ``model_params``. If
        ``preprocess_methods`` or ``detection_validators`` is provided as
        a list of dicts, each entry is resolved into a callable.

        Args:
            overrides: A dict mapping field names to new values.
        """
        valid_names = {f.name for f in fields(self)}
        for key, value in overrides.items():
            if key in valid_names:
                if key == "preprocess_methods":
                    value = [
                        resolve_callable_descriptor(m, default_module=image_preprocessing)
                        if isinstance(m, dict)
                        else m
                        for m in value
                    ]
                elif key == "detection_validators":
                    value = [resolve_callable_descriptor(v) if isinstance(v, dict) else v for v in value]
                setattr(self, key, value) # TODO: Remove, non-pythonic
            else:
                self.model_params[key] = value

        self.validate()

    def to_dict(self) -> dict:
        """Convert the config to a JSON-serializable dict.

        Delegates to ``SerializableClass.to_dict`` for the trivial fields,
        then rewrites the two callable-valued list fields into JSON-safe
        descriptor lists via ``_serialize_callable_descriptor``.

        Returns:
            A plain dict representation of this config.
        """
        data = super().to_dict()
        data["preprocess_methods"] = [serialize_callable_descriptor(m) for m in self.preprocess_methods]
        data["detection_validators"] = [serialize_callable_descriptor(v) for v in self.detection_validators]
        return data

    @classmethod
    def from_dict(cls, raw_dict: dict):
        """Create an OCRConfig from a plain dict.

        Resolves the two callable-valued list fields into live callables
        via ``_resolve_callable_descriptor`` and then defers to
        ``SerializableClass.from_dict`` for instantiation. Keys that do
        not correspond to a dataclass field are silently dropped (besides
        ``_type``), allowing config JSONs to carry extra metadata such as
        ``alias``.

        Args:
            raw_dict: Dict with at least ``model_name`` and optionally
                ``model_params``, ``detection_validators``, and
                ``preprocess_methods``.

        Returns:
            An OCRConfig instance.

        Raises:
            TypeError: If ``model_name`` is missing from ``raw_dict``.
        """
        valid_names = {f.name for f in fields(cls)}
        prepared = {k: v for k, v in raw_dict.items() if k == TYPE_KEY or k in valid_names}

        raw_pp = prepared.get("preprocess_methods")
        if raw_pp is not None:
            prepared["preprocess_methods"] = [
                resolve_callable_descriptor(m, default_module=image_preprocessing)
                if isinstance(m, dict)
                else m
                for m in raw_pp
            ]

        raw_validators = prepared.get("detection_validators")
        if raw_validators is not None:
            prepared["detection_validators"] = [
                resolve_callable_descriptor(v) if isinstance(v, dict) else v for v in raw_validators
            ]

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
