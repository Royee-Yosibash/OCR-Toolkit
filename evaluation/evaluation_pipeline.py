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

import copy
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np

from evaluation.metrics import Metric
from evaluation.ocr_ground_truth import OCRGroundTruth
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_backbone.ocr_result import OCRResult
from utils.dataset_utils import dataset_generator
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


class StatsDict(TypedDict):
    """Schema for a single metric's summary statistics.

    Produced by :meth:`EvaluationResult._compute_stats` and stored as the
    leaf value in the aggregate JSON.

    Attributes:
        mean: Sample mean of the metric values.
        ci: Confidence intervals keyed by level (as string, e.g. ``"95"``),
            each mapping to ``[lower, upper]``.
        n: Number of values in the sample.
        min: Minimum value.
        max: Maximum value.
        median: Median value.
    """

    mean: float
    ci: dict[str, list[float]]
    n: int
    min: float
    max: float
    median: float


# Dynamic-key levels of the aggregate JSON. Keys are user-defined strings
# (metric class name, OCR id, dataset tag), so plain dict aliases are used
# rather than TypedDicts.
MetricStatsDict = dict[str, StatsDict]  # metric_name -> stats
OCRMetricsDict = dict[str, MetricStatsDict]  # ocr_id -> metric_name -> stats
AggregateData = dict[str, OCRMetricsDict]  # tag -> ocr_id -> metric_name -> stats


@dataclass(frozen=True)
class TagBreakdown:
    """One tag's slice of an aggregate result.

    Bundles the per-tag sub-dict together with the OCR-id and metric-name
    lists derived from its top-level keys, so consumers can iterate without
    re-deriving them.

    Attributes:
        data: The ``ocr_id -> metric_name -> stats`` sub-dict for the
            chosen tag, shaped as :data:`OCRMetricsDict`.
    """

    data: OCRMetricsDict

    @property
    def labels(self) -> list[str]:
        return list(self.data.keys())

    @property
    def metrics(self) -> list[str]:
        return list(next(iter(self.data.values())).keys())


@dataclass(frozen=True)
class AggregateResult:
    """Typed wrapper around the aggregate-results JSON of an evaluation run.

    The underlying JSON shape is ``tag -> ocr_id -> metric_name -> stats``.
    This class adds named per-tag access without otherwise changing the
    structure, so it round-trips losslessly via :meth:`to_dict` /
    :class:`AggregateResult(data=...)`.

    Construction paths:
        - ``AggregateResult(data=raw_dict)`` — wrap an in-memory dict (used
          internally by the evaluation pipeline).
        - ``AggregateResult.from_path(results_dir)`` — load from disk.

    Attributes:
        data: The raw nested dict, preserved as-is for JSON round-trips
            and for callers that want full structural access. Conforms to
            :data:`AggregateData` (``tag -> ocr_id -> metric_name ->`` :class:`StatsDict`).
    """

    data: AggregateData

    @classmethod
    def from_path(cls, results_dir: str | Path) -> "AggregateResult":
        """Load an aggregate JSON from an evaluation output directory.

        Args:
            results_dir: Path to a directory containing
                ``AGGREGATE_RESULTS_FILE`` (``aggregate.json``).

        Returns:
            An :class:`AggregateResult` wrapping the parsed JSON.

        Raises:
            FileNotFoundError: If the aggregate file does not exist.
        """
        return cls(data=load_json(Path(results_dir) / AGGREGATE_RESULTS_FILE))

    @property
    def tags(self) -> list[str]:
        """Return the tag keys available in this aggregate.

        Returns:
            The top-level keys of the underlying dict, including
            ``ALL_TAGS_KEY`` ("all") and one entry per dataset tag.
        """
        return list(self.data.keys())

    def for_tag(self, name: str = ALL_TAGS_KEY) -> TagBreakdown:
        """Return the `TagBreakdown` for a single tag.

        Args:
            name: Tag key to slice on. Defaults to ``ALL_TAGS_KEY``
                ("all"), i.e. the across-all-images aggregate.

        Returns:
            A `TagBreakdown` bundling the tag sub-dict, its OCR-id
            labels, and the metric names present.

        Raises:
            KeyError: If ``name`` is not present. The error message lists
                the available tags.
        """
        if name not in self.data:
            raise KeyError(f"Tag {name!r} not found in aggregate. Available tags: {sorted(self.data.keys())}")
        return TagBreakdown(data=self.data[name])

    def to_dict(self) -> AggregateData:
        """Return an independent nested dict representation (for JSON serialization).

        Returns:
            A deep copy of the underlying data, conforming to
            :data:`AggregateData`: ``tag -> ocr_id -> metric_name ->`` :class:`StatsDict`.
            Safe for callers to mutate without affecting this instance.
        """
        return copy.deepcopy(self.data)


@dataclass
class EvaluationResult:
    """Container for evaluation output.

    Args:
        per_image: Nested dict of ocr_id -> metric_name -> image_id -> value.
        image_tags: Dict mapping image_id to the ground truth tags for that image.
        aggregate: An :class:`AggregateResult` wrapping the nested
            ``tag -> ocr_id -> metric_name -> stats`` structure. The key
            ``"all"`` contains aggregate stats across all images; each
            tag key contains stats for images with that tag.
    """

    per_image: dict[str, dict[str, dict[int, float]]]  #  TODO: Change name
    image_tags: dict[int, list[str]]
    aggregate: AggregateResult = field(init=False, default=None)

    def __post_init__(self):
        """Validate inputs and compute aggregate statistics.

        Raises:
            ValueError: If ``image_tags`` is empty.
        """
        if not self.image_tags:
            raise ValueError("image_tags must not be empty")
        self.aggregate = self._build_aggregate_result(self.per_image, self.image_tags)
        logger.info("Aggregate statistics computed successfully")

    @classmethod
    def _build_aggregate_result(
        cls,
        per_image: dict[str, dict[str, dict[int, float]]],
        image_tags: dict[int, list[str]],
        ci_levels: tuple[int, ...] = DEFAULT_CI_LEVELS,
    ) -> AggregateResult:
        """Compute aggregate statistics from per-image results.

        Computes stats across all images under the ``"all"`` key, and
        per-tag stats under each tag key. Uses Beta-distribution CIs for
        bounded metrics and bootstrap CIs for unbounded metrics.

        Args:
            per_image: Nested dict of ocr_id -> metric_name -> image_id -> value.
            image_tags: Dict mapping image_id to the ground truth tags.
            ci_levels: Confidence interval percentages to compute. For each
                level *L*, the ``ci`` dict stores ``L`` -> [lower, upper].

        Returns:
            An :class:`AggregateResult` wrapping the :data:`AggregateData`
            dict shaped as ``tag -> ocr_id -> metric_name ->`` :class:`StatsDict`.

        Raises:
            ValueError: If any tag group contains fewer than 2 images.
        """
        all_tags = {tag for tags in image_tags.values() for tag in tags}
        tag_groups = {ALL_TAGS_KEY: set(image_tags.keys())}
        for tag in all_tags:
            tag_groups[tag] = {img_id for img_id, tags in image_tags.items() if tag in tags}
        logger.info(
            f"Computing aggregate statistics for {len(tag_groups)} tag group(s): {list(tag_groups.keys())}",
        )

        insufficient_tags = {tag: len(ids) for tag, ids in tag_groups.items() if len(ids) < 2}
        if insufficient_tags:
            raise ValueError(f"{insufficient_tags}: fewer than 2 images cannot produce meaningful statistics")

        aggregate: AggregateData = {}
        for tag_key, image_ids in tag_groups.items():
            aggregate[tag_key] = {}
            for ocr_id, metric_dict in per_image.items():
                aggregate[tag_key][ocr_id] = {}
                for metric_name, image_scores in metric_dict.items():
                    values = np.array([v for img_id, v in image_scores.items() if img_id in image_ids])
                    if len(values):
                        aggregate[tag_key][ocr_id][metric_name] = cls._compute_stats(
                            values,
                            metric_name,
                            ci_levels,
                        )
        return AggregateResult(data=aggregate)

    @staticmethod
    def _compute_stats(
        values: np.ndarray,
        metric_name: str,
        ci_levels: tuple[int, ...],
    ) -> StatsDict:
        """Compute summary statistics for a set of metric values.

        Args:
            values: Array of metric values.
            metric_name: Name of the metric, used to select the CI method.
            ci_levels: Confidence interval percentages to compute.

        Returns:
            A :class:`StatsDict` with mean, ci, n, min, max, and median.
        """
        ci_fn = beta_ci if Metric._registry[metric_name].is_bounded else bootstrap_ci
        ci: dict[str, list[float]] = {}
        for level in ci_levels:
            ci[str(level)] = ci_fn(values, level)
        return StatsDict(
            mean=float(np.mean(values)),
            ci=ci,
            n=len(values),
            min=float(np.min(values)),
            max=float(np.max(values)),
            median=float(np.median(values)),
        )

    def save_results_to_file(self, output_dir: Path):
        """Save aggregate results to a JSON file.

        Args:
            output_dir: Directory to save the aggregate file in.
        """
        save_json(output_dir / AGGREGATE_RESULTS_FILE, self.aggregate.to_dict())


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
        dataset: Dataset iterable, used when metrics_only is False. Ignored
            (and may be None) when metrics_only is True.
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
            logger.debug("Skipping %s -- cached result exists", ocr_id)
            continue
        logger.info("Running OCR engine %s", ocr_id)
        result = ocr.get_text_detections(image=image)
        save_json(save_path / OCR_RESULTS_FILE, result.to_dict(), mkdir=True)
        logger.info("Saved OCR result for %s to %s", ocr_id, save_path)


def evaluation_pipeline(
    metrics: list[Metric],
    output_dir: str | Path,
    dataset: str | Path | None = None,
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
        dataset: Path to a local directory containing ``images/`` and
            ``ground_truth/`` subdirectories. Required unless metrics_only
            is True.
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
        ValueError: If labels is provided but its length does not match ocrs,
            or if ocrs/dataset are missing when metrics_only is False.
    """
    if not metrics_only:
        if ocrs is None:
            raise ValueError("ocrs must be provided when metrics_only is False.")
        if dataset is None:
            raise ValueError("dataset must be provided when metrics_only is False.")
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

    iterator_dataset = None if metrics_only else dataset_generator(dataset)
    for image_id, image, ground_truth in _build_iterator(output_dir, iterator_dataset, metrics_only):
        image_dir = output_dir / IMAGE_DIR_NAME.format(x=image_id)
        image_tags[image_id] = ground_truth.tags
        logger.info("Processing image %s (tags=%s)", image_id, ground_truth.tags)
        if not metrics_only:
            image_dir.mkdir(parents=True, exist_ok=True)
            save_json(image_dir / GT_FILE, ground_truth.to_dict())
            run_multiple_ocrs_and_save(image=image, ocrs=ocrs, ocr_ids=labels, save_dir=image_dir, overwrite=overwrite)

        _score_image(image_dir, ground_truth, metrics, image_id, per_image)

    logger.info(
        "Evaluation loop complete -- scored %d image(s) across %d OCR engine(s)",
        len(image_tags),
        len(per_image),
    )
    evaluation_result = EvaluationResult(per_image=per_image, image_tags=image_tags)
    evaluation_result.save_results_to_file(output_dir=output_dir)
    logger.info("Aggregate results saved to %s", output_dir)
    return evaluation_result
