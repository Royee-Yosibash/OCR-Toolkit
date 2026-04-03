"""Record expected OCR results for all registered modules.

For each registered OCR module, runs OCR on the given image(s) and saves
the output to <output_dir>/<module>/expected/<image_stem>.json.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from consts import IMAGE_EXTENSIONS
from evaluation.evaluation_pipeline import run_multiple_ocrs_and_save
from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_modules import import_all_modules
from utils.datasets_handles import collect_images
from utils.json_utils import load_json


def record_with_all_ocr_modules(
    image_path: str,
    output_dir: str,
    config: dict,
) -> None:
    """Run all registered OCR modules on image(s) and save expected results.

    Args:
        image_path: Path to a single image or a directory of images.
        output_dir: Root directory where per-module results are saved.
        config: Dict of extra OCR config fields (e.g. grid_rows,
            model_params). Must not contain ``model_name``.
    """
    import_all_modules()
    
    if not OCRAbstract._registry:
        raise NotImplementedError("No OCR modules found.")

    if "model_name" in config:
        raise ValueError("The config must not set a model_name")

    images = collect_images(Path(image_path))
    if not images:
        raise IOError(f"No images found at {image_path}")

    ocrs, ocr_ids = list(), list()
    for name, cls in OCRAbstract._registry.items():
        ocr_ids.append(name)
        ocrs.append(OCRAbstract.from_config(config=OCRConfig(model_name=name, **config)))

    for img_path in images:
        image = np.array(Image.open(img_path))
        image_save_dir = Path(output_dir) / img_path.stem
        image_save_dir.mkdir(parents=True, exist_ok=True)
        run_multiple_ocrs_and_save(image=image, ocrs=ocrs, ocr_ids=ocr_ids,
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
