"""End-to-end OCR evaluation pipeline.

Runs one or more OCR engines on a dataset, computes metrics per image,
persists results to disk, and produces aggregate statistics. Supports
re-running metrics on previously saved OCR results without re-running
inference.

Directory structure created by evaluate::

    output_dir/
        image_0/
            EasyOCRModule_0/
                ocr_result.json
                metrics.json
            EasyOCRModule_1/
                ocr_result.json
                metrics.json
        image_1/
            ...
        aggregate.json
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from evaluation.metrics import Metric, METRICS_BOUNDED_LOOKUP
from evaluation.ocr_ground_truth import OCRGroundTruth
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_backbone.ocr_result import OCRResult
from utils.datasets_handles import dataset_generator
from utils.json_utils import save_json, load_json
from utils.statistics import beta_ci, bootstrap_ci

logger = logging.getLogger(__name__)

IMAGE_DIR_NAME = "image_{x}"
OCR_RESULTS_FILE = "ocr_result.json"
AGGREGATE_RESULTS_FILE = "aggregate.json"
GT_FILE = "ground_truth.json"
METRICS_FILE = "metrics.json"
DEFAULT_CI_LEVELS = (95,)


def run_multiple_ocrs_and_save(image: np.ndarray, 
                               ocrs: list[OCRAbstract], 
                               labels: list[str], 
                               save_dir: Path,
                               overwrite=False):
    """Run multiple OCR engines on an image and persist each result to disk.

    Args:
        image: Input image as a numpy array (H x W x C).
        ocrs: List of initialized OCR engine instances.
        labels: List of label strings, one per OCR engine, used as
            subdirectory names under ``save_dir``.
        save_dir: Directory where per-engine results are saved.
        overwrite: If True, re-run OCR even when a saved result already
            exists for that engine.
    """
    for ocr, label in zip(ocrs, labels):
        save_path = save_dir / label
        if (save_path / OCR_RESULTS_FILE).exists() and not overwrite:
            continue
        result = ocr.get_text_bb(image=image)
        save_json(save_path / OCR_RESULTS_FILE, result.to_dict(), mkdir=True)


@dataclass
class EvaluationResult:
    """Container for evaluation output.

    Args:
        per_image: Nested dict of ocr_label -> metric_name -> image_id -> value.
        aggregate: Nested dict of ocr_label -> metric_name -> stat_name -> value.
            Stats include mean, std, min, max, median.
    """

    per_image: dict[str, dict[str, dict[int, float]]] = field(default_factory=dict)
    aggregate: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._compute_aggregate()

    def _compute_aggregate(self,ci_levels: tuple[int, ...] = DEFAULT_CI_LEVELS,):
        """Compute aggregate statistics from per-image results.

        Uses Beta-distribution CIs for bounded metrics and bootstrap CIs
        for unbounded metrics.

        Args:
            ci_levels: Confidence interval percentages to compute. For each
                level *L*, the ``ci`` dict stores ``L`` -> [lower, upper].

        Returns:
            None
        """
        aggregate = {}
        for ocr_label, metric_dict in self.per_image.items():
            aggregate[ocr_label] = {}
            for metric_name, image_scores in metric_dict.items():
                values = np.array(list(image_scores.values()))
                is_bounded = METRICS_BOUNDED_LOOKUP.get(metric_name, False)
                ci_fn = beta_ci if is_bounded else bootstrap_ci
                ci = {}
                for level in ci_levels:
                    ci[str(level)] = ci_fn(values, level)
                aggregate[ocr_label][metric_name] = {
                    "mean": float(np.mean(values)),
                    "ci": ci,
                    "n": len(values),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }

        self.aggregate = aggregate

    def save_to_file(self, output_dir: Path):
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

        for metric_name, value in metrics_dict.items():
            per_image.setdefault(label, {}).setdefault(metric_name, {})[image_id] = value


def evaluation_pipeline(
    metrics: list[Metric],
    output_dir: str | Path,
    dataset: str | None = None,
    ocrs: list[OCRAbstract] | None = None,
    overwrite: bool = False,
    metrics_only: bool = False,
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

    Returns:
        An EvaluationResult with per-image scores and aggregate stats.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image: dict[str, dict[str, dict[int, float]]] = {}
    labels = [f"{type(ocr).__name__}_{i}" for i, ocr in enumerate(ocrs)] if ocrs else []

    for image_id, image, ground_truth in \
        _build_iterator(output_dir, dataset_generator(dataset), metrics_only):
        
        image_dir = output_dir / IMAGE_DIR_NAME.format(x=image_id)
        if not metrics_only:
            image_dir.mkdir(parents=True, exist_ok=True)
            save_json(image_dir / GT_FILE, ground_truth.to_dict())
            run_multiple_ocrs_and_save(image=image, ocrs=ocrs, labels=labels, 
                               save_dir=image_dir, overwrite=overwrite)

        _score_image(image_dir, ground_truth, metrics, image_id, per_image)

    evaluation_result = EvaluationResult(per_image=per_image)
    evaluation_result.save_to_file(output_dir=output_dir)
    return evaluation_result
