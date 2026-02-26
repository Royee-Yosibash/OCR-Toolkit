import numpy as np

from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_abstract import OCRAbstact
from ocr_backbone.ocr_result import OCRResult
from pathlib import Path

from utils.json_utils import save_json

from datasets import load_dataset


def run_and_save(image: np.ndarray, ocr: OCRAbstact, save_path: str) -> OCRResult:
    """Run OCR on an image and save the results to a JSON file.

    The OCR configuration is read from ``ocr.config``.

    Args:
        image: Input image as a numpy array (H x W x C).
        ocr: An initialized OCR instance with its config already set.
        save_path: Path where the results JSON will be saved.

    Returns:
        The OCRResult produced by the OCR run.
    """
    result = ocr.get_text_bb(image)
    save_json(save_path, result.to_dict())
    return result


def run_ocr_on_dataset(
    dataset_name: str,
    config: OCRConfig,
    output_dir: str,
    subset: str = "test",
    ):
    """Load a HuggingFace dataset and run OCR on every image.

    Each result is saved as ``<index>.json`` inside ``output_dir``.

    Args:
        dataset_name: HuggingFace dataset identifier
            (e.g. "getomni-ai/ocr-benchmark").
        config: OCR configuration used to initialise the engine.
        output_dir: Directory where per-image result JSON files are saved.
        subset: Dataset split to load. Defaults to "test".

    Returns:
        A list of OCRResult objects, one per dataset image.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split=subset)
    ocr = OCRAbstact.from_config(config)

    for idx, item in enumerate(dataset):
        image = np.array(item["image"].convert("RGB"))
        save_path = str(output_path / f"{idx}.json")
        run_and_save(image, ocr, save_path)

