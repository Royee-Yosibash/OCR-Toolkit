"""Utilities for loading images and tags from the local dataset directory."""

from collections.abc import Generator
from pathlib import Path

import numpy as np
from PIL import Image

from consts import APP_ROOT, IMAGE_EXTENSIONS
from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr_result import OCRResult
from utils.json_utils import load_json

DATASET_DIR = APP_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
TAGS_DIR = DATASET_DIR / "tags"


def collect_images(path: Path) -> list[Path]:
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


def load_image(image_path: Path) -> np.ndarray:
    """Load an image file as a numpy array in RGB format.

    Args:
        image_path: Path to the image file.

    Returns:
        Image as a numpy array (H x W x 3).
    """
    return np.array(Image.open(image_path).convert("RGB"))


def load_tags(tags_path: Path) -> OCRResult:
    """Load a tags JSON file as an OCRResult.

    Args:
        tags_path: Path to the JSON tags file.

    Returns:
        An OCRResult parsed from the JSON file.
    """
    data = load_json(tags_path)
    return OCRResult.from_dict(data)


def _find_image_for_stem(stem: str) -> Path:
    """Find an image file in the images directory matching the given stem.

    Args:
        stem: Filename stem to search for (without extension).

    Returns:
        The Path to the matching image.
    """
    matches = [IMAGES_DIR / f"{stem}{ext}" for ext in IMAGE_EXTENSIONS
               if (IMAGES_DIR / f"{stem}{ext}").exists()]
    return matches[0]


def validate_tags(tags_path: Path, image_path: Path) -> None:
    """Validate a tags JSON file against its corresponding image.

    Checks that every bounding box is structurally valid and that
    coordinates fall within the image dimensions. All errors are
    aggregated and raised together.

    Args:
        tags_path: Path to the JSON tags file.
        image_path: Path to the corresponding image.

    Raises:
        ValueError: If any validation errors are found. The message
            contains all errors separated by newlines.
    """
    errors = []

    data = load_json(tags_path)

    img = Image.open(image_path)
    img_w, img_h = img.size

    for i, bb_data in enumerate(data["bounding_boxes"]):
        prefix = f"BB #{i + 1}"
        try:
            bb = BoundingBox.from_dict(bb_data)
        except (ValueError, KeyError, TypeError) as e:
            errors.append(f"{prefix}: invalid bounding box: {e}")
            continue

        (x1, y1), (x2, y2) = bb.coordinates
        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
            errors.append(
                f"{prefix}: coordinates ({x1},{y1})-({x2},{y2}) "
                f"exceed image bounds ({img_w}x{img_h})"
            )

    if errors:
        raise ValueError(
            f"Validation failed for {tags_path.name} "
            f"({len(errors)} error(s)):\n" + "\n".join(errors)
        )


def validate_dataset() -> None:
    """Validate the entire dataset directory.

    Checks that every image has a corresponding tags file and vice
    versa, then validates each tag file against its image.

    Raises:
        FileNotFoundError: If any images are missing tags or tags are
            missing images.
        ValueError: If any tag validation errors are found across the
            dataset. The message contains all errors grouped by stem.
    """
    image_stems = set()
    for ext in IMAGE_EXTENSIONS:
        for img_path in IMAGES_DIR.glob(f"*{ext}"):
            image_stems.add(img_path.stem)

    tag_stems = {p.stem for p in TAGS_DIR.glob("*.json")}

    missing_tags = sorted(image_stems - tag_stems)
    if missing_tags:
        raise FileNotFoundError(
            f"Missing tags for {len(missing_tags)} image(s): "
            + ", ".join(missing_tags)
        )

    missing_images = sorted(tag_stems - image_stems)
    if missing_images:
        raise FileNotFoundError(
            f"Missing images for {len(missing_images)} tag(s): "
            + ", ".join(missing_images)
        )

    all_errors: dict[str, list[str]] = {}
    for tags_path in sorted(TAGS_DIR.glob("*.json")):
        stem = tags_path.stem
        image_path = _find_image_for_stem(stem)
        try:
            validate_tags(tags_path, image_path)
        except ValueError as e:
            all_errors[stem] = str(e).split("\n")[1:]

    if all_errors:
        lines = []
        for stem in sorted(all_errors):
            lines.append(f"{stem}:")
            for err in all_errors[stem]:
                lines.append(f"  {err}")
        raise ValueError(
            f"Validation errors in {len(all_errors)} file(s):\n"
            + "\n".join(lines)
        )


def dataset_generator() -> Generator[tuple[np.ndarray, OCRResult, str], None, None]:
    """Yield (image, ground_truth, stem) tuples from the dataset.

    Iterates over all tags files in the dataset, finds matching images,
    and yields them one at a time. Suitable for use as input + ground
    truth when testing an OCR engine.

    Yields:
        A tuple of (image, ocr_result, stem) where image is a numpy
        array (H x W x 3), ocr_result is the ground-truth OCRResult,
        and stem is the shared filename stem.

    """
    for tags_path in sorted(TAGS_DIR.glob("*.json")):
        stem = tags_path.stem
        image_path = _find_image_for_stem(stem)
        image = load_image(image_path)
        tags = load_tags(tags_path)
        yield image, tags, stem
