"""Utilities for loading images and ground_truth from the local dataset directory."""

from collections.abc import Generator
from pathlib import Path

import numpy as np
from PIL import Image

from consts import APP_ROOT, IMAGE_EXTENSIONS
from evaluation.ocr_ground_truth import OCRGroundTruth
from ocr_backbone.polygon import Polygon
from utils.json_utils import load_json

DATASET_DIR = APP_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
TAGS_DIR = DATASET_DIR / "ground_truth"


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
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
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


def load_ground_truth(tags_path: Path) -> OCRGroundTruth:
    """Load a ground_truth JSON file as an OCRGroundTruth.

    Args:
        tags_path: Path to the JSON ground_truth file.

    Returns:
        An OCRGroundTruth parsed from the JSON file.
    """
    data = load_json(tags_path)
    return OCRGroundTruth.from_dict(data)


def _find_image_for_stem(stem: str, images_dir: Path | None = None) -> Path:
    """Find an image file in a directory matching the given stem.

    Args:
        stem: Filename stem to search for (without extension).
        images_dir: Directory to search in. Defaults to the built-in
            dataset images directory.

    Returns:
        The Path to the matching image.

    Raises:
        FileNotFoundError: If no image with a supported extension exists
            for the given stem.
    """
    if images_dir is None:
        images_dir = IMAGES_DIR
    candidates = (images_dir / f"{stem}{ext}" for ext in IMAGE_EXTENSIONS)
    try:
        return next(p for p in candidates if p.exists())
    except StopIteration:
        raise FileNotFoundError(f"No image found for stem '{stem}' in {images_dir}") from None


def validate_tags(tags_path: Path, image_path: Path) -> None:
    """Validate a ground_truth JSON file against its corresponding image.

    Checks that every bounding box is structurally valid and that
    coordinates fall within the image dimensions. All errors are
    aggregated and raised together.

    Args:
        tags_path: Path to the JSON ground_truth file.
        image_path: Path to the corresponding image.

    Raises:
        ValueError: If any validation errors are found. The message
            contains all errors separated by newlines.
    """
    errors = []

    data = load_json(tags_path)

    img = Image.open(image_path)
    img_w, img_h = img.size

    for i, det_data in enumerate(data["detections"]):
        prefix = f"Detection #{i + 1}"
        try:
            bb = Polygon.create(det_data)
        except (ValueError, KeyError, TypeError) as e:
            errors.append(f"{prefix}: invalid detection: {e}")
            continue

        for crds in bb.coordinates:
            if crds[0] > img_w or crds[1] > img_h:
                errors.append(f"{prefix}: coordinates ({crds[0]},{crds[1]}) exceed image bounds ({img_w}x{img_h})")

    if errors:
        raise ValueError(f"Validation failed for {tags_path.name} ({len(errors)} error(s)):\n" + "\n".join(errors))


def validate_dataset(dataset_root: Path | str | None = None) -> None:
    """Validate a dataset directory.

    Checks that every image has a corresponding ground_truth file and vice
    versa, then validates each tag file against its image.

    Args:
        dataset_root: Root directory of the dataset, expected to contain
            ``images/`` and ``ground_truth/`` subdirectories. Defaults to
            the built-in ``dataset/`` directory when *None*.

    Raises:
        FileNotFoundError: If the dataset root or its required
            subdirectories do not exist, or if any images are missing
            ground_truth or ground_truth are missing images.
        ValueError: If any tag validation errors are found across the
            dataset. The message contains all errors grouped by stem.
    """
    root = Path(dataset_root) if dataset_root is not None else DATASET_DIR
    images_dir = root / "images"
    gt_dir = root / "ground_truth"
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory does not exist: {images_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"ground truth directory does not exist: {gt_dir}")

    image_stems = set()
    for ext in IMAGE_EXTENSIONS:
        for img_path in images_dir.glob(f"*{ext}"):
            image_stems.add(img_path.stem)

    tag_paths = sorted(gt_dir.glob("*.json"))
    tag_stems = {p.stem for p in tag_paths}

    missing_tags = sorted(image_stems - tag_stems)
    if missing_tags:
        raise FileNotFoundError(f"Missing ground_truth for {len(missing_tags)} image(s): " + ", ".join(missing_tags))

    missing_images = sorted(tag_stems - image_stems)
    if missing_images:
        raise FileNotFoundError(f"Missing images for {len(missing_images)} tag(s): " + ", ".join(missing_images))

    all_errors: dict[str, list[str]] = {}
    for tags_path in tag_paths:
        stem = tags_path.stem
        image_path = _find_image_for_stem(stem, images_dir)
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
        raise ValueError(f"Validation errors in {len(all_errors)} file(s):\n" + "\n".join(lines))


def dataset_generator(
    dataset_root: Path | str | None = None,
) -> Generator[tuple[np.ndarray, OCRGroundTruth], None, None]:
    """Yield (image, ground_truth) tuples from a dataset directory.

    The directory must contain an ``images/`` subfolder with image files
    and a ``ground_truth/`` subfolder with identically-stemmed JSON tag files.

    Args:
        dataset_root: Root directory of the dataset. Defaults to the
            built-in ``dataset/`` directory when *None*.

    Yields:
        A tuple of (image, ground_truth) where image is a numpy array
        (H x W x 3) and ground_truth is the ground-truth OCRGroundTruth.
        :rtype: Generator[tuple[np.ndarray, OCRGroundTruth], None, None]
    """
    root = Path(dataset_root) if dataset_root is not None else DATASET_DIR
    images_dir = root / "images"
    gt_dir = root / "ground_truth"
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory does not exist: {images_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"ground truth directory does not exist: {gt_dir}")

    for gt_path in sorted(gt_dir.glob("*.json")):
        stem = gt_path.stem
        image = load_image(_find_image_for_stem(stem, images_dir))
        gt = load_ground_truth(gt_path)
        yield image, gt
