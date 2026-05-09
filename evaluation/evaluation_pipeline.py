"""End-to-end OCR evaluation pipeline.

Runs one or more OCR engines on a dataset, computes metrics per image,
persists results to disk, and produces aggregate statistics. Supports
re-running metrics on previously saved OCR results without re-running
inference.

Directory structure created by evaluate:

    output_dir/
        aggregate.json
        image_0/
            EasyOCRModule_0/
                ocr_result.json
                metrics.json
            EasyOCRModule_1/
                ocr_result.json
                metrics.json
        image_1/
            ...
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from evaluation.metrics import METRICS_BOUNDED_LOOKUP, Metric
from evaluation.ocr_ground_truth import OCRGroundTruth
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_backbone.ocr_result import OCRResult
from utils.datasets_handles import dataset_generator
from utils.json_utils import load_json, save_json
from utils.statistics import beta_ci, bootstrap_ci

logger = logging.getLogger(__name__)

IMAGE_DIR_NAME = "image_{x}"
OCR_RESULTS_FILE = "ocr_result.json"
AGGREGATE_RESULTS_FILE = "aggregate.json"
GT_FILE = "ground_truth.json"
METRICS_FILE = "metrics.json"
DEFAULT_CI_LEVELS = (95,)
ALL_TAGS_KEY = "all"


@dataclass
class EvaluationResult:
    """Container for evaluation output.

    Args:
        per_image: Nested dict of ocr_id -> metric_name -> image_id -> value.
        image_tags: Dict mapping image_id to the ground truth tags for that image.
        aggregate: Nested dict of tag -> ocr_id -> metric_name -> stats.
            The key ``"all"`` contains aggregate stats across all images.
            Each tag key contains stats for images with that tag.
    """

    per_image: dict[str, dict[str, dict[int, float]]] = field(default_factory=dict)
    image_tags: dict[int, list[str]] = field(default_factory=dict)
    aggregate: dict[str, dict[str, dict[str, dict]]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        assert self.image_tags, "image_tags must not be empty"
        self._compute_aggregate()

    def _compute_aggregate(self, ci_levels: tuple[int, ...] = DEFAULT_CI_LEVELS):
        """Compute aggregate statistics from per-image results.

        Computes stats across all images under the ``"all"`` key, and
        per-tag stats under each tag key. Uses Beta-distribution CIs for
        bounded metrics and bootstrap CIs for unbounded metrics.

        Args:
            ci_levels: Confidence interval percentages to compute. For each
                level *L*, the ``ci`` dict stores ``L`` -> [lower, upper].

        Returns:
            None
        """

        all_tags = {tag for tags in self.image_tags.values() for tag in tags}
        tag_groups = {ALL_TAGS_KEY: set(self.image_tags.keys())}
        for tag in all_tags:
            tag_groups[tag] = {img_id for img_id, tags in self.image_tags.items() if tag in tags}
        logger.info(
            f"Computing aggregate statistics for {len(tag_groups)} tag group(s): {list(tag_groups.keys())}",
        )

        insufficient_tags = {tag: len(ids) for tag, ids in tag_groups.items() if len(ids) < 2}
        if insufficient_tags:
            raise ValueError(f"Tags with fewer than 2 images cannot produce meaningful statistics: {insufficient_tags}")

        aggregate = {}
        for tag_key, image_ids in tag_groups.items():
            aggregate[tag_key] = {}
            for ocr_id, metric_dict in self.per_image.items():
                aggregate[tag_key][ocr_id] = {}
                for metric_name, image_scores in metric_dict.items():
                    values = np.array([v for img_id, v in image_scores.items() if img_id in image_ids])
                    if len(values):
                        aggregate[tag_key][ocr_id][metric_name] = self._compute_stats(
                            values,
                            metric_name,
                            ci_levels,
                        )

        self.aggregate = aggregate
        logger.info("Aggregate statistics computed successfully")

    @staticmethod
    def _compute_stats(
        values: np.ndarray,
        metric_name: str,
        ci_levels: tuple[int, ...],
    ) -> dict:
        """Compute summary statistics for a set of metric values.

        Args:
            values: Array of metric values.
            metric_name: Name of the metric, used to select the CI method.
            ci_levels: Confidence interval percentages to compute.

        Returns:
            A dict with mean, ci, n, min, max, and median.
        """
        is_bounded = METRICS_BOUNDED_LOOKUP.get(metric_name, False)
        ci_fn = beta_ci if is_bounded else bootstrap_ci
        ci = {}
        for level in ci_levels:
            ci[str(level)] = ci_fn(values, level)
        return {
            "mean": float(np.mean(values)),
            "ci": ci,
            "n": len(values),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

    def save_results_to_file(self, output_dir: Path):
        """Save aggregate results to a JSON file.

        Args:
            output_dir: Directory to save the aggregate file in.
        """
        save_json(output_dir / AGGREGATE_RESULTS_FILE, self.aggregate)


def _compute_metrics(
    prediction: OCRResult,
    ground_truth: OCRGroundTruth,
    metrics: list[Metric],
) -> dict[str, float]:
    """Run all metric functions on a prediction/ground-truth pair.

    Args:
        prediction: The predicted OCR result.
        ground_truth: The reference ground truth.
        metrics: List of metric callables.

    Returns:
        A dict mapping metric class name to its computed value.
    """
    return {type(metric).__name__: metric(prediction, ground_truth) for metric in metrics}


def _build_iterator(
    output_dir: Path,
    dataset: Iterable[tuple[np.ndarray, OCRGroundTruth]] | None,
    metrics_only: bool,
):
    """Build the image iterator for the evaluation loop.

    Args:
        output_dir: Root output directory.
        dataset: Dataset iterable, used when metrics_only is False.
        metrics_only: If True, iterate saved directories instead of dataset.

    Yields:
        Tuples of (image_id, image_or_none, ground_truth).
    """
    if metrics_only:
        saved_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir())
        for i, d in enumerate(saved_dirs):
            yield i, None, OCRGroundTruth.from_dict(load_json(d / GT_FILE))
    else:
        for image_id, (image, gt) in enumerate(dataset):
            yield image_id, image, gt


def _score_image(
    image_dir: Path,
    ground_truth: OCRGroundTruth,
    metrics: list[Metric],
    image_id: int,
    per_image: dict[str, dict[str, dict[int, float]]],
) -> None:
    """Compute and save metrics for all OCR results in an image directory.

    Args:
        image_dir: Directory containing OCR subdirectories.
        ground_truth: Ground truth OCRGroundTruth.
        metrics: List of metric callables.
        image_id: Index of the current image.
        per_image: Accumulator dict, mutated in place.
    """
    for ocr_dir in sorted(image_dir.iterdir()):
        if not ocr_dir.is_dir() or not (ocr_dir / OCR_RESULTS_FILE).exists():
            continue
        label = ocr_dir.name
        prediction = OCRResult.from_dict(load_json(ocr_dir / OCR_RESULTS_FILE))
        metrics_dict = _compute_metrics(prediction, ground_truth, metrics)
        save_json(ocr_dir / METRICS_FILE, metrics_dict)
        logger.debug(
            "Image %d | %s metrics: %s",
            image_id,
            label,
            {k: f"{v:.4f}" for k, v in metrics_dict.items()},
        )

        for metric_name, value in metrics_dict.items():
            per_image.setdefault(label, {}).setdefault(metric_name, {})[image_id] = value


def run_multiple_ocrs_and_save(
    image: np.ndarray, ocrs: list[OCRAbstract], ocr_ids: list[str], save_dir: Path, overwrite=False
):
    """Run multiple OCR engines on an image and persist each result to disk.

    Args:
        image: Input image as a numpy array (H x W x C).
        ocrs: List of initialized OCR engine instances.
        ocr_ids: List of id strings, one per OCR engine, used as
            subdirectory names under ``save_dir``.
        save_dir: Directory where per-engine results are saved.
        overwrite: If True, re-run OCR even when a saved result already
            exists for that engine.
    """
    for ocr, ocr_id in zip(ocrs, ocr_ids, strict=True):
        save_path = save_dir / ocr_id
        if (save_path / OCR_RESULTS_FILE).exists() and not overwrite:
            logger.debug(f"Skipping {ocr_id} -- cached result exists")
            continue
        logger.info("Running OCR engine ")
        result = ocr.get_text_detections(image=image)
        save_json(save_path / OCR_RESULTS_FILE, result.to_dict(), mkdir=True)
        logger.info(f"Saved OCR result for {ocr_id} to {save_path}")


def evaluation_pipeline(
    metrics: list[Metric],
    output_dir: str | Path,
    dataset: str | None = None,
    ocrs: list[OCRAbstract] | None = None,
    overwrite: bool = False,
    metrics_only: bool = False,
    labels: list[str] | None = None,
) -> EvaluationResult:
    """Run OCR evaluation on a dataset and persist results.

    For each image, runs every OCR engine, computes all metrics against
    the ground truth, and saves results to disk. When metrics_only is
    True, skips inference and recomputes from saved results.

    Args:
        metrics: List of metric callables with signature
            (OCRResult, OCRResult) -> float | int | bool.
        output_dir: Directory where results are persisted.
        dataset: Dataset identifier string. Currently, a path to a local
            directory containing ``images/`` and ``ground_truth/`` subdirectories.
            Required unless metrics_only is True.
        ocrs: List of initialized OCR instances to evaluate. Required
            unless metrics_only is True.
        overwrite: If True, re-run OCR even when saved results exist.
        metrics_only: If True, skip OCR inference and recompute metrics
            from previously saved results.
        labels: Optional list of display names for each OCR engine, one
            per entry in ``ocrs``. When None, labels are auto-generated
            as ``"{ClassName}_{i}"``.

    Returns:
        An EvaluationResult with per-image scores and aggregate stats.

    Raises:
        ValueError: If labels is provided but its length does not match ocrs.
    """
    if labels is not None and ocrs is not None and len(labels) != len(ocrs):
        raise ValueError(f"labels length ({len(labels)}) must match ocrs length ({len(ocrs)}).")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image: dict[str, dict[str, dict[int, float]]] = {}
    image_tags: dict[int, list[str]] = {}
    if labels is None:
        labels = [ocr.alias if ocr.alias else f"{type(ocr).__name__}_{i}" for i, ocr in enumerate(ocrs)] if ocrs else []

    mode = "metrics-only" if metrics_only else "full"
    logger.info(
        "Starting evaluation pipeline (mode=%s, output_dir=%s, ocr_engines=%s)",
        mode,
        output_dir,
        labels,
    )

    for image_id, image, ground_truth in _build_iterator(output_dir, dataset_generator(dataset), metrics_only):
        image_dir = output_dir / IMAGE_DIR_NAME.format(x=image_id)
        image_tags[image_id] = ground_truth.tags
        logger.info(f"Processing image {image_id} (tags={ground_truth.tags})")
        if not metrics_only:
            image_dir.mkdir(parents=True, exist_ok=True)
            save_json(image_dir / GT_FILE, ground_truth.to_dict())
            run_multiple_ocrs_and_save(image=image, ocrs=ocrs, ocr_ids=labels, save_dir=image_dir, overwrite=overwrite)

        _score_image(image_dir, ground_truth, metrics, image_id, per_image)

    logger.info(
        f"Evaluation loop complete -- scored {len(image_tags)} image(s) across {len(per_image)} OCR engine(s)",
    )
    evaluation_result = EvaluationResult(per_image=per_image, image_tags=image_tags)
    evaluation_result.save_results_to_file(output_dir=output_dir)
    logger.info(f"Aggregate results saved to {output_dir}")
    return evaluation_result
