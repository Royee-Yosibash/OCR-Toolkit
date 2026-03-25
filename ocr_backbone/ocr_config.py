import importlib
from collections.abc import Callable
from dataclasses import dataclass, fields, field
from pathlib import Path

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
        preprocess_methods: A list of preprocessing step descriptors. Each
            entry is a dict with "name" (a function name in
            image_preprocessing) and optional "kwargs" to pass to it.
    """

    model_name: str
    model_params: dict = field(default_factory=dict)
    bb_validator: Callable[[BoundingBox], bool] | None = None
    preprocess_methods: list[dict] = field(default_factory=list)

    def update(self, overrides: dict) -> None:
        """Update config attributes from a dict.

        Only keys that correspond to existing dataclass fields are applied.
        Unknown keys are ignored.

        Args:
            overrides: A dict mapping field names to new values.
        """
        valid_names = {f.name for f in fields(self)}
        for key, value in overrides.items():
            if key in valid_names:
                setattr(self, key, value)
            else:
                self.model_params[key] = value
    
    
    def to_dict(self) -> dict:
        """Convert the config to a JSON-serializable dict.

        ``bb_validator`` is stored as a ``"module.path:function_name"``
        string when present, so the dict can be round-tripped through JSON.

        Returns:
            A plain dict representation of this config.
        """
        data = {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "preprocess_methods": self.preprocess_methods,
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

        Args:
            raw_dict: Dict with at least ``model_name`` and optionally
                ``model_params``, and ``bb_validator``.

        Returns:
            An OCRConfig instance.
        """
        bb_validator = raw_dict.get("bb_validator", None)
        if isinstance(bb_validator, str):
            module_path, attr_name = bb_validator.rsplit(":", 1)
            module = importlib.import_module(module_path)
            bb_validator = getattr(module, attr_name)


        return cls(
            model_name=raw_dict["model_name"],
            model_params=raw_dict.get("model_params", {}),
            bb_validator=bb_validator,
            preprocess_methods=raw_dict.get("preprocess_methods", []),
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
    
