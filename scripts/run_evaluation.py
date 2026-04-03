"""Run EasyOCR on the dataset and evaluate against ground truth.

Loads every image/tag pair from the dataset, runs all registered OCR
modules, computes CER, WER, word recall, word precision, and word
count ratio, and writes per-image and aggregate results to
``scripts/results/``.

Usage::

    python -m scripts.run_evaluation [--output-dir scripts/results]
"""

import argparse
import logging
from pathlib import Path

from ocr_modules import import_all_modules
from evaluation.evaluation_pipeline import evaluation_pipeline, ALL_TAGS_KEY
from evaluation.metrics import (
    ocr_result_cer,
    ocr_result_wer,
    word_count_ratio,
    word_precision,
    word_recall,
)
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_backbone.ocr_config import OCRConfig
from evaluation.evaluation_analysis import load_aggregate, plot_metric_comparison, plot_radar, plot_stat_range
from utils.datasets_handles import DATASET_DIR

ALL_METRICS = [
    ocr_result_cer,
    ocr_result_wer,
    word_count_ratio,
    word_precision,
    word_recall,
]

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments and run the evaluation pipeline.

    Run from the terminal as::

        python -m scripts.run_evaluation [--dataset dataset] [--output-dir scripts/results]
    """
    parser = argparse.ArgumentParser(
        description="Run OCR evaluation on the dataset with all registered modules.",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_DIR,
        help="Path to a dataset root directory containing images/ and ground_truth/ "
             "(default: dataset).",
    )
    parser.add_argument(
        "--output-dir",
        default="scripts/results",
        help="Directory for saving evaluation results (default: scripts/results).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    import_all_modules()

    logger.info(f"Registered OCR modules: {list(OCRAbstract._registry.keys())}")
    ocrs = []
    for name in OCRAbstract._registry:
        logger.info(f"Instantiating OCR module: {name}")
        ocrs.append(OCRAbstract.from_config(OCRConfig(model_name=name)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Running evaluation (dataset={args.dataset}, " + \
        "output_dir={output_dir}," + \
        f"metrics={[type(m).__name__ for m in ALL_METRICS]})"
    )

    evaluation_pipeline(
        metrics=ALL_METRICS,
        output_dir=output_dir,
        dataset=args.dataset,
        ocrs=ocrs,
        overwrite=True,
    )

    logger.info(f"Aggregate results saved to {output_dir}")

    agg = load_aggregate(output_dir)[ALL_TAGS_KEY]
    logger.info("Generating plots")
    plot_metric_comparison(agg, save_path=output_dir / 'metric_comparison.png')
    plot_radar(agg, save_path=output_dir / 'radar.png')
    plot_stat_range(agg, save_path=output_dir / 'stat_range.png')
    logger.info(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
