import json
from collections.abc import Callable
from dataclasses import dataclass, fields, field
from pathlib import Path

from ocr_backbone.bounding_box import BoundingBox


@dataclass
class OCRConfig:
    """Configuration for an OCR run.

    Args:
        model_name: Name of the OCR model to use.
        model_params: Model-specific runtime parameters.
        grid_rows: Number of rows to split the image into.
        grid_cols: Number of columns to split the image into.
        bb_validator: Optional function that takes a BoundingBox and returns
            True if the bounding box is valid. Invalid bounding boxes are
            discarded after OCR inference.
    """

    model_name: str
    model_params: dict = field(default_factory=dict)
    grid_rows: int = 1
    grid_cols: int = 1
    bb_validator: Callable[[BoundingBox], bool] | None = None

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
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return OCRConfig(
        model_name=data["model_name"],
        model_params=data.get("model_params", {}),
        grid_rows=data.get("grid_rows", 1),
        grid_cols=data.get("grid_cols", 1),
    )
