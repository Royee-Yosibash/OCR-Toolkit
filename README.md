# OCR-Toolkit

A unified framework for integrating, benchmarking, and comparing OCR engines
under controlled conditions. Unlike ad-hoc OCR scripts that couple a single
engine to a single preprocessing pipeline, OCR-Toolkit decouples the engine,
the preprocessing pipeline, and the evaluation metrics into interchangeable
components. Integrating a new OCR engine requires one subclass with one method;
the framework handles image preprocessing, coordinate remapping, metric
computation, and result persistence automatically.

## Introduction

Selecting the optimal OCR engine and configuration for a given input type is
largely trial and error. Different engines, preprocessing pipelines, and
parameter choices can yield vastly different results, yet there is no
standardized method for quantifying those differences across a dataset.
OCR-Toolkit replaces intuition with data: execute the pipeline, examine the
metrics, and let confidence intervals, per-tag breakdowns, and side-by-side
plots identify the optimal combination.

## Use Cases

- Compare multiple OCR engines on the same dataset under identical conditions.
- Measure the impact of a preprocessing step (e.g. contour splitting vs. grid
  splitting) on recognition accuracy.
- Integrate a new OCR engine with minimal boilerplate and immediately benchmark
  it against existing solutions.
- Create and maintain ground-truth annotations using the included tagging tool.
- Make data-driven decisions about OCR configuration rather than relying on
  manual inspection.

## Features

- **Unified OCR interface** -- integrate any OCR engine by subclassing
  `OCRAbstract`. Ships with EasyOCR, PaddleOCR, and pytesseract
  implementations.
- **Modular preprocessing** -- composable preprocessing pipeline supporting
  grid splitting, contour-based segmentation, binarization (adaptive and Otsu),
  and user-supplied steps. Combine stages via configuration or code to build the
  pipeline that fits your input data.
- **Evaluation pipeline** -- compute character accuracy, word accuracy, word
  recall, word precision, and word count ratio against ground-truth
  annotations, with per-image and aggregate results including confidence
  intervals and supporting visualizations
- **Tagging tool** -- browser-based Flask UI for creating and editing
  ground-truth bounding-box annotations.

## Versioning

Current version: **0.1.4** (released 2026-05-05).

## Installation

Requires Python 3.12+.

The project ships three independent optional dependency groups on top of the
always-installed base. Pick any subset, combine them, or use `full` for
everything.

**Base** (core abstractions, evaluation primitives, no engines, no app):

```bash
pip install -e .
```

**OCRs** (all OCR engines: EasyOCR, PaddleOCR, PaddlePaddle, pytesseract):

```bash
pip install -e ".[ocrs]"
```

> Note: `pytesseract` also requires the system `tesseract` binary, e.g.
> `sudo apt-get install -y tesseract-ocr` on Debian/Ubuntu.

**Dataset Tagging App** (Flask UI plus its end-to-end test deps):

```bash
pip install -e ".[dataset-tagging-app]"
```

**Developer** (ruff, pytest, pre-commit):

```bash
pip install -e ".[dev]"
pre-commit install
```

**Full** (all three groups above):

```bash
pip install -e ".[full]"
```

Combinations are supported, e.g. `pip install -e ".[ocrs,dev]"`.

All code style is enforced by [Ruff](https://docs.astral.sh/ruff/) via
pre-commit hooks. After `pre-commit install`, every commit is automatically
checked (lint + format). The ruff configuration lives in `pyproject.toml`
under `[tool.ruff]`.

## Project Structure

```
ocr_backbone/       Core abstractions: OCRAbstract, OCRConfig, OCRResult, BoundingBox, InputImage
ocr_modules/        Concrete OCR engine implementations (EasyOCR, PaddleOCR, pytesseract)
evaluation/         Evaluation pipeline, metrics, ground truth, and plotting utilities
utils/              Shared utilities (JSON I/O, dataset loading, statistics, serialization)
scripts/            CLI tools: run_evaluation, draw_bboxes, tagging tool
dataset/            Local dataset with images/ and ground_truth/ subdirectories
tests/              Unit, regression, and application tests
```

## Usage

### Run Evaluation

Execute all registered OCR modules against the dataset and generate metrics
and plots:

```bash
python -m scripts.run_evaluation [--config config.json] [--dataset path/to/dataset] [--output-dir scripts/results]
```

- `--config` accepts a single JSON config file or a directory of JSON configs.
  Without it, all registered OCR modules are evaluated with default settings.
- `--dataset` defaults to the built-in `dataset/` directory. The directory must
  contain `images/` and `ground_truth/` subdirectories with matching stems.
- `--output-dir` defaults to `scripts/results`.

Results are saved as JSON files and PNG plots (bar chart, radar, range plot).

### Draw Bounding Boxes

Overlay persisted OCR results on an image:

```bash
python -m scripts.draw_bboxes <image_path> <results_json>
```

Produces `<image_stem>_with_bounding_box.<ext>` in the same directory.

### Tagging Tool

Launch the browser-based annotation UI:

```bash
python -m scripts.image_ocr_tagging_tool [--port 5000]
```

Opens `http://localhost:5000` in the default browser.

## Configuration

OCR runs are configured via `OCRConfig`, which can be loaded from a JSON file.
A minimal configuration requires only `model_name` and `model_params`:

```json
{
    "model_name": "EasyOCRModule",
    "model_params": {}
}
```

A full configuration can include preprocessing steps and a bounding-box
validator:

```json
{
    "model_name": "EasyOCRModule",
    "model_params": {"languages": ["en"]},
    "preprocess_methods": [
      {"name": "contour_split_image"}
    ]
}
```

Preprocessing method names are resolved from `ocr_backbone.image_preprocessing`
by default. Dotted paths (e.g. `"my_package.module.func"`) are dynamically
imported. Optional `"kwargs"` are bound via `functools.partial`.

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

## License

MIT -- see [LICENSE](LICENSE).
