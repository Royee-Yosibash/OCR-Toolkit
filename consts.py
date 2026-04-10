import json
from pathlib import Path

APP_ROOT = Path(__file__).parent

with open(APP_ROOT / "release_data.json") as _f:
    __version__ = json.load(_f)["version"]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
