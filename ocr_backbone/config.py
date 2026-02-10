import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OCRConfig:
    """Configuration for an OCR run.

    Args:
        model_name: Name of the OCR model to use.
        model_params: Model-specific runtime parameters.
        grid_rows: Number of rows to split the image into.
        grid_cols: Number of columns to split the image into.
    """

    model_name: str
    model_params: dict = field(default_factory=dict)
    grid_rows: int = 1
    grid_cols: int = 1


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
