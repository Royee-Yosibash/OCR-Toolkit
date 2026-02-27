"""Utilities for reading and writing JSON files."""

import json
from pathlib import Path


def save_json(path: str | Path, data: dict, mkdir: bool = False) -> None:
    """Save a dict to a JSON file.

    Args:
        path: Path to the output JSON file.
        data: Dict to serialize.
        mkdir: If True, create parent directories as needed.
    """
    path = Path(path)
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict:
    """Load a dict from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed dict.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))
