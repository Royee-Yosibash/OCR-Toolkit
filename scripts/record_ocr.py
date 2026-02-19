"""Record expected OCR results for all registered modules.

For each registered OCR module, runs OCR on the given image(s) and saves
the output to <output_dir>/<module>/expected/<image_stem>.json.
"""

import argparse
import importlib
import pkgutil
from pathlib import Path

import numpy as np
from PIL import Image

import ocr_modules
from ocr_backbone.ocr_config import OCRConfig, load_config
from ocr_backbone.ocr_abstract import OCRAbstact
from utils.run_and_save import run_and_save

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def _import_all_modules() -> None:
    """Import every module inside the ocr_modules package to trigger
    subclass registration in OCRAbstact._registry.
    """
    package_path = Path(ocr_modules.__file__).parent
    for finder, name, is_pkg in pkgutil.iter_modules([str(package_path)]):
        importlib.import_module(f"ocr_modules.{name}")


def _collect_images(path: Path) -> list[Path]:
    """Return a sorted list of image paths from a file or directory.

    Args:
        path: A path to a single image file or a directory containing
            images.

    Returns:
        A sorted list of image file paths.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is a file with an unsupported extension.
    """
    if path.is_dir():
        images = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        return images
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    return [path]


def _resolve_output_dir(cls: type, output_root: Path) -> Path:
    """Return the expected-results directory for a registered OCR class.

    Maps the class back to its source module name (e.g.
    ``ocr_modules.easyocr_module`` -> ``easyocr``) and returns
    ``<output_root>/<name>/expected/``.

    Args:
        cls: The OCR subclass.
        output_root: Root directory for all module outputs.

    Returns:
        Path to the expected-results directory.
    """
    module_name = cls.__module__.rsplit(".", 1)[-1]
    short_name = module_name.removesuffix("_module")
    output_dir = output_root / short_name / "expected"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def record_all(
    image_path: str,
    output_dir: str,
    config: OCRConfig | None = None,
) -> None:
    """Run all registered OCR modules on image(s) and save expected results.

    Args:
        image_path: Path to a single image or a directory of images.
        output_dir: Root directory where per-module results are saved.
        config: Optional OCR config. When None a default config is
            created per module using the registered class name.
    """
    _import_all_modules()

    if not OCRAbstact._registry:
        print("No OCR modules found.")
        return

    images = _collect_images(Path(image_path))
    if not images:
        print(f"No images found at {image_path}")
        return

    output_root = Path(output_dir)

    for name, cls in OCRAbstact._registry.items():
        module_config = config or OCRConfig(model_name=name)
        module_output = _resolve_output_dir(cls, output_root)

        print(f"Recording {name}...")
        for img_path in images:
            image = np.array(Image.open(img_path))
            save_path = str(module_output / f"{img_path.stem}.json")
            result = run_and_save(image, cls(config=module_config), save_path)
            print(f"  {img_path.name}: {len(result.bounding_boxes)} bounding boxes -> {save_path}")


def main() -> None:
    """Parse arguments and record OCR output for all modules.

    Run from the terminal as::

        python -m scripts.record_ocr <image_path> <output_dir> [--config <config.json>]

    ``image_path`` may be a single image file or a directory of images.
    ``output_dir`` is the root folder where results are saved, organized
    as ``<output_dir>/<module>/expected/<image_stem>.json``.
    """
    parser = argparse.ArgumentParser(
        description="Record expected OCR results for all registered modules.",
    )
    parser.add_argument("image_path", help="Path to an image file or a directory of images.")
    parser.add_argument("output_dir", help="Root directory for saving results.")
    parser.add_argument(
        "--config", "-c",
        help="Path to a JSON config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    record_all(args.image_path, args.output_dir, config)


if __name__ == "__main__":
    main()
