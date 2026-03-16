"""Run EasyOCR on the dataset and evaluate against ground truth.

Loads every image/tag pair from the dataset, runs all registered OCR
modules, computes CER, WER, word recall, word precision, and word
count ratio, and writes per-image and aggregate results to
``scripts/results/``.

Usage::

    python -m scripts.run_evaluation [--output-dir scripts/results]
"""

import argparse
import importlib
import pkgutil
from pathlib import Path
import ocr_modules
from evaluation.evaluation_pipeline import evaluation_pipeline
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


ALL_METRICS = [
    ocr_result_cer,
    ocr_result_wer,
    word_count_ratio,
    word_precision,
    word_recall,
]

DEFAULT_DATASET = "dataset"


def _import_all_modules() -> None:
    """Import every module inside the ocr_modules package to trigger
    subclass registration in OCRAbstract._registry.
    """
    package_path = Path(ocr_modules.__file__).parent
    for _, name, _ in pkgutil.iter_modules([str(package_path)]):
        importlib.import_module(f"ocr_modules.{name}")


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
        default=DEFAULT_DATASET,
        help="Path to a dataset root directory containing images/ and tags/ "
             "(default: dataset).",
    )
    parser.add_argument(
        "--output-dir",
        default="scripts/results",
        help="Directory for saving evaluation results (default: scripts/results).",
    )
    args = parser.parse_args()

    _import_all_modules()

    ocrs = []
    for name in OCRAbstract._registry:
        ocrs.append(OCRAbstract.from_config(OCRConfig(model_name=name)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluation_pipeline(
        metrics=ALL_METRICS,
        output_dir=output_dir,
        dataset=args.dataset,
        ocrs=ocrs,
        overwrite=True,
    )

    print(f"\nAggregate results saved to {output_dir}/aggregate.json")

    agg = load_aggregate(output_dir)
    plot_metric_comparison(agg, save_path=output_dir / 'metric_comparison.png')
    plot_radar(agg, save_path=output_dir / 'radar.png')
    plot_stat_range(agg, save_path=output_dir / 'stat_range.png')


if __name__ == "__main__":
    main()
