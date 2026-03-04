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
from consts import IMAGE_EXTENSIONS
from evaluation.evaluation_pipeline import run_multiple_ocrs_and_save, run_ocr_and_save
from ocr_backbone.ocr_config import OCRConfig, load_config
from ocr_backbone.ocr_abstract import OCRAbstact
from utils.datasets_handles import collect_images
from utils.json_utils import load_json


def _import_all_modules() -> None:
    """Import every module inside the ocr_modules package to trigger
    subclass registration in OCRAbstact._registry.
    """
    package_path = Path(ocr_modules.__file__).parent
    for finder, name, is_pkg in pkgutil.iter_modules([str(package_path)]):
        importlib.import_module(f"ocr_modules.{name}")


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


def record_with_all_ocr_modules(
    image_path: str,
    output_dir: str,
    config: dict,
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
        raise NotImplementedError("No OCR modules found.")

    if "model_name" in config:
        raise IOError(f"The config can not set a model_name")

    images = collect_images(Path(image_path))
    if not images:
        raise IOError(f"No images found at {image_path}")

    ocrs, labels = list(), list()
    for name, cls in OCRAbstact._registry.items():
        labels.append(name)
        ocrs.append(OCRAbstact.from_config(config=OCRConfig(model_name=name, **config)))

    for img_path in images:
        image = np.array(Image.open(img_path))
        image_save_dir = Path(output_dir) / img_path.stem
        image_save_dir.mkdir(parents=True, exist_ok=True)
        run_multiple_ocrs_and_save(image=image, ocrs=ocrs, labels=labels, 
                                   save_dir=image_save_dir, overwrite=True)
        
        # Compute metrics too? we can then join with evaluate....


def main() -> None:
    """Parse arguments and record outputs from all OCR modules using the same config.

    Run from the terminal as::

        python -m scripts.record_ocrs_same_config <image_path> <output_dir> [--config <config.json>]

    ``image_path`` may be a single image file or a directory of images.
    ``output_dir`` is the root folder where results are saved, organized
    as ``<output_dir>/<module>/expected/<image_stem>.json``.
    """
    parser = argparse.ArgumentParser(
        description="Record expected OCR results for all registered modules using the same config.",
    )
    parser.add_argument("image_path", help="Path to an image file or a directory of images.")
    parser.add_argument("output_dir", help="Root directory for saving results.")
    parser.add_argument("--config", "-c", help="Path to a JSON config file.",)
    args = parser.parse_args()

    config = load_json(args.config) if args.config else {}
    record_with_all_ocr_modules(args.image_path, args.output_dir, config)


if __name__ == "__main__":
    main()
