import importlib
from collections.abc import Callable
from dataclasses import dataclass, fields, field
from functools import partial
from pathlib import Path

import ocr_backbone.image_preprocessing as image_preprocessing
from ocr_backbone.bounding_box import BoundingBox
from utils.json_utils import load_json


@dataclass
class OCRConfig:
    """Configuration for an OCR run.

    Args:
        model_name: Name of the OCR model to use.
        model_params: Model-specific runtime parameters.
        bb_validator: Optional function that takes a BoundingBox and returns
            True if the bounding box is valid. Invalid bounding boxes are
            discarded after OCR inference.
        preprocess_methods: A list of preprocessing callables. Each callable
            accepts an InputImage and returns an InputImage or a list of
            InputImages. When constructed via ``from_dict``, method
            descriptors (dicts with "name" and optional "kwargs") are
            resolved into callables automatically.
    """

    model_name: str
    model_params: dict = field(default_factory=dict)
    bb_validator: Callable[[BoundingBox], bool] | None = None
    preprocess_methods: list[Callable] = field(default_factory=list)

    @staticmethod
    def _resolve_pp_method(pp_method: dict) -> Callable:
        """Resolve a preprocessing method descriptor into a callable.

        If the name contains a dot it is treated as a fully qualified
        dotted path (e.g. ``"my_package.module.func"``).  The last
        segment is the attribute name and everything before it is the
        module path that will be dynamically imported.  Otherwise the
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
                    value = [
                        self._resolve_pp_method(m) if isinstance(m, dict) else m
                        for m in value
                    ]
                setattr(self, key, value)
            else:
                self.model_params[key] = value


    def to_dict(self) -> dict:
        """Convert the config to a JSON-serializable dict.

        ``bb_validator`` is stored as a ``"module.path:function_name"``
        string when present, so the dict can be round-tripped through JSON.
        Each preprocessing callable is serialized back to a dict with
        ``"name"`` as a fully qualified dotted path.

        Returns:
            A plain dict representation of this config.
        """
        serialized_pp = []
        for method in self.preprocess_methods:
            func = method.func if isinstance(method, partial) else method
            kwargs = method.keywords if isinstance(method, partial) else {}
            name = f"{func.__module__}.{func.__qualname__}"
            entry: dict = {"name": name}
            if kwargs:
                entry["kwargs"] = kwargs
            serialized_pp.append(entry)

        data = {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "preprocess_methods": serialized_pp,
        }
        if self.bb_validator is not None:
            module = self.bb_validator.__module__
            qualname = self.bb_validator.__qualname__
            data["bb_validator"] = f"{module}:{qualname}"
        return data

    @classmethod
    def from_dict(cls, raw_dict: dict):
        """Create an OCRConfig from a plain dict.

        ``bb_validator`` may be a callable or a dotted-path string in the
        form ``"module.path:function_name"``. Strings are dynamically
        imported.

        ``preprocess_methods`` entries that are dicts are resolved into
        callables via ``_resolve_pp_method``.

        Args:
            raw_dict: Dict with at least ``model_name`` and optionally
                ``model_params``, ``bb_validator``, and
                ``preprocess_methods``.

        Returns:
            An OCRConfig instance.
        """
        bb_validator = raw_dict.get("bb_validator", None)
        if isinstance(bb_validator, str):
            module_path, attr_name = bb_validator.rsplit(":", 1)
            module = importlib.import_module(module_path)
            bb_validator = getattr(module, attr_name)

        raw_pp = raw_dict.get("preprocess_methods", [])
        preprocess_methods = [
            cls._resolve_pp_method(m) if isinstance(m, dict) else m
            for m in raw_pp
        ]

        return cls(
            model_name=raw_dict["model_name"],
            model_params=raw_dict.get("model_params", {}),
            bb_validator=bb_validator,
            preprocess_methods=preprocess_methods,
        )



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
    data = load_json(path)
    return OCRConfig.from_dict(data)
    
