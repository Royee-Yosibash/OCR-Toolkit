# OCR-Highlight

An OCR enhancement toolset that wraps multiple OCR engines behind a unified
interface and provides an evaluation pipeline for comparing their performance
on custom datasets.

## Features

- **Unified OCR interface** -- plug in any OCR engine by subclassing
  `OCRAbstract`. Ships with EasyOCR and PaddleOCR modules.
- **Grid-based inference** -- split images into configurable grids, run OCR
  per cell, and remap bounding boxes back to original coordinates.
- **Evaluation pipeline** -- compute CER, WER, word recall, word precision,
  and word count ratio against ground-truth tags, with per-image and aggregate
  results including confidence intervals.
- **Visualization** -- bar charts, radar plots, and range plots for comparing
  OCR modules side by side.
- **Tagging tool** -- browser-based UI for creating and editing ground-truth
  bounding box annotations.

## Project Structure

```
ocr_backbone/       Core abstractions: OCRAbstract, OCRConfig, OCRResult, BoundingBox
ocr_modules/        Concrete OCR engine implementations (EasyOCR, PaddleOCR)
evaluation/         Evaluation pipeline, metrics, and analysis/plotting utilities
utils/              Shared utilities (JSON I/O, dataset loading, statistics)
scripts/            CLI tools: run_evaluation, draw_bboxes, record_ocrs, tagging tool
dataset/            Local dataset with images/ and tags/ subdirectories
tests/              Unit and regression tests
```

## Installation

Requires Python 3.12+.

**Basic** (core functionality without OCR engine dependencies):

```bash
pip install -e .
```

**Full** (includes EasyOCR, PaddleOCR, and PaddlePaddle):

```bash
pip install -e ".[full]"
```

## Usage

### Run Evaluation

Run all registered OCR modules against the dataset and generate metrics and
plots:

```bash
python -m scripts.run_evaluation [--dataset dataset] [--output-dir scripts/results]
```

Results are saved to the output directory as JSON files and PNG plots.

### Draw Bounding Boxes

Overlay saved OCR results on an image:

```bash
python -m scripts.draw_bboxes <image_path> <results_json>
```

### Record OCR Results

Run all registered OCR modules on image(s) with a shared config and save the
raw results:

```bash
python -m scripts.record_ocrs_same_config <image_path> <output_dir> [--config config.json]
```

### Tagging Tool

Launch the browser-based annotation UI:

```bash
python -m scripts.image_ocr_tagging_tool [--port 5000]
```

## Configuration

OCR runs are configured via `OCRConfig`, which can be loaded from a JSON file:

```json
{
    "model_name": "EasyOCRModule",
    "model_params": {"languages": ["en"]},
    "grid_rows": 1,
    "grid_cols": 1
}
```

See `default_config.json` for the minimal template.

## Adding a New OCR Module

Subclass `OCRAbstract` and implement `_run_single`:

```python
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_backbone.ocr_result import OCRResult

class MyOCRModule(OCRAbstract):
    def __init__(self, config):
        super().__init__(config)
        # initialize your engine

    def _run_single(self, image, single_run_model_params):
        # run inference, return OCRResult
        ...
```

The subclass is auto-registered by class name and becomes available to
`OCRAbstract.from_config` and the evaluation pipeline.

## Tests

```bash
python -m pytest tests/
```

Or via the built-in runner:

```bash
python -m tests.tests_runner
```

## License

MIT -- see [LICENSE](LICENSE).
