# OCR-Toolkit

Choosing the right OCR engine and configuration for a given document type is
largely trial and error. Different engines, preprocessing pipelines, and
parameter choices can produce vastly different results, and there is no
standard way to measure that difference across a dataset.

OCR-Highlight solves this by providing a single simple framework where you can plug
in any OCR engine, define any image preprocessing pipelines (e.g. grid splitting,
contour-based segmentation, binarization, and more), and run a reproducible
evaluation against ground-truth annotations. It computes standard metrics
(CER, WER, word recall, word precision), generates comparison plots, and
stores every result so you can iterate on configurations without losing
previous runs.

OCR-Toolkit is equal parts a framework for standardizing OCR pipelines and a
benchmarking tool for measuring their output. Adding a new OCR engine takes a
single subclass with one method to implement -- the framework handles image
preprocessing, coordinate remapping, metric computation, and result
persistence automatically. This makes it practical to onboard any OCR
solution, whether open-source or proprietary, and immediately benchmark it
against every other engine on the same dataset. The same applies to
preprocessing strategies: swap a grid split for a contour-based segmentation,
point the evaluation at a config file, and get a direct, quantitative answer
on whether it helps.

The goal is to replace intuition with data. Instead of guessing which engine
works best for a document type, or whether a preprocessing step improves
accuracy, you run the pipeline and read the numbers. Confidence intervals,
per-tag breakdowns, and side-by-side plots make it straightforward to identify
which combination of engine, preprocessing, and parameters performs best for
your specific use case.

Use it when you need to:

- Compare multiple OCR engines on the same dataset under identical conditions.
- Measure how a preprocessing step (e.g. contour splitting vs. grid splitting)
  affects recognition accuracy.
- Onboard a new OCR engine with minimal boilerplate and immediately benchmark
  it against existing solutions.
- Build and maintain ground-truth annotations with the included tagging tool.
- Make data-driven decisions about OCR configuration instead of relying on
  manual inspection.

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
python -m scripts.run_evaluation [--config config.json] [--dataset dataset] [--output-dir scripts/results]
```

Results are saved to the output directory as JSON files and PNG plots.

### Draw Bounding Boxes

Overlay saved OCR results on an image:

```bash
python -m scripts.draw_bboxes <image_path> <results_json>
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
