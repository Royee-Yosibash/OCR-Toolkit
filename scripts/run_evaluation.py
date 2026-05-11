"""Run OCR evaluation on a dataset and compare against ground truth.

Loads every image/tag pair from the dataset, runs OCR modules, computes
CER, WER, word recall, word precision, and word count ratio, and writes
per-image and aggregate results to ``scripts/results/``.

When ``--config`` is provided, only the module specified in that config
file is evaluated (with its preprocessing pipeline, model params, etc.).
Without ``--config``, all registered OCR modules are evaluated with
default settings.

Usage::

    python -m scripts.run_evaluation [--config config.json] [--output-dir scripts/results]
"""

import argparse
import logging
from pathlib import Path

from evaluation.evaluation_analysis import load_aggregate, plot_metric_comparison, plot_radar, plot_stat_range
from evaluation.evaluation_pipeline import ALL_TAGS_KEY, evaluation_pipeline
from evaluation.metrics import (
    ocr_result_char_accuracy,
    ocr_result_word_accuracy,
    word_count_ratio,
    word_precision,
    word_recall,
)
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_backbone.ocr_config import OCRConfig, load_config
from ocr_modules import import_all_modules
from utils.dataset_utils import DATASET_DIR

ALL_METRICS = [
    ocr_result_char_accuracy,
    ocr_result_word_accuracy,
    word_count_ratio,
    word_precision,
    word_recall,
]

logger = logging.getLogger(__name__)


def evaluate() -> None:
    """Parse arguments and run the evaluation pipeline.

    Run from the terminal as::

        python -m scripts.run_evaluation [--config config.json] [--dataset dataset] [--output-dir scripts/results]
    """
    parser = argparse.ArgumentParser(
        description="Run OCR evaluation on the dataset.",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Path to a JSON config file or a directory of JSON config files. "
        "Each config specifies a module and its settings to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_DIR,
        help="Path to a dataset root directory containing images/ and ground_truth/ (default: dataset).",
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

    labels = None
    if args.config:
        config_path = Path(args.config)
        if config_path.is_dir():
            config_files = sorted(config_path.glob("*.json"))
            if not config_files:
                raise FileNotFoundError(f"No JSON config files found in {config_path}")
            logger.info(f"Loading {len(config_files)} config(s) from {config_path}")
        else:
            config_files = [config_path]
        ocrs = []
        labels = []
        for cf in config_files:
            config = load_config(cf)
            logger.info(f"Loaded config from {cf} (model_name={config.model_name})")
            ocrs.append(OCRAbstract.from_config(config))
            labels.append(cf.stem)
    else:
        registered = OCRAbstract.registered_models()
        logger.info(f"Registered OCR modules: {registered}")
        ocrs = []
        for name in registered:
            logger.info(f"Instantiating OCR module: {name}")
            ocrs.append(OCRAbstract.from_config(OCRConfig(model_name=name)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Running evaluation (dataset={args.dataset}, "
        f"output_dir={output_dir}, "
        f"metrics={[type(m).__name__ for m in ALL_METRICS]})"
    )

    evaluation_pipeline(
        metrics=ALL_METRICS,
        output_dir=output_dir,
        dataset=args.dataset,
        ocrs=ocrs,
        overwrite=True,
        labels=labels,
    )

    agg = load_aggregate(output_dir)[ALL_TAGS_KEY]
    logger.info("Generating plots")
    plot_metric_comparison(agg, save_path=output_dir / "metric_comparison.png")
    plot_radar(agg, save_path=output_dir / "radar.png")
    plot_stat_range(agg, save_path=output_dir / "stat_range.png")
    logger.info(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    evaluate()
