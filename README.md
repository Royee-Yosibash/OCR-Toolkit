# OCR-Toolkit

Selecting the optimal OCR engine and configuration for a given input type is
largely trial and error. Different engines, preprocessing pipelines, and
parameter choices can yield vastly different results, yet there is no
standardized method for quantifying those differences across a dataset.

OCR-Toolkit addresses this by providing a unified framework for integrating
any OCR engine, defining composable image preprocessing pipelines (e.g. grid
splitting, contour-based segmentation, binarization), and executing
reproducible evaluations against ground-truth annotations. It computes
standard metrics (CER, WER, word recall, word precision), generates
comparison plots, and persists all results to support iterative configuration
refinement without data loss.

The toolkit serves both as a framework for standardizing OCR pipelines and
as a benchmarking suite for measuring their output. Integrating a new OCR
engine requires a single subclass with one method -- the framework handles
image preprocessing, coordinate remapping, metric computation, and result
persistence automatically. This makes it practical to onboard any OCR
solution, whether open-source or proprietary, and immediately benchmark it
against all other engines on the same dataset. The same principle applies to
preprocessing strategies: substitute a grid split for a contour-based
segmentation, reference a configuration file, and obtain a direct,
quantitative comparison.

The objective is to replace intuition with data. Rather than estimating which
engine performs best for a given input type, or whether a preprocessing step
improves accuracy, execute the pipeline and examine the results. Confidence
intervals, per-tag breakdowns, and side-by-side plots provide clear evidence
for identifying the optimal combination of engine, preprocessing, and
parameters for a specific use case.

### Use Cases

- Compare multiple OCR engines on the same dataset under identical conditions.
- Measure the impact of a preprocessing step (e.g. contour splitting vs. grid
  splitting) on recognition accuracy.
- Integrate a new OCR engine with minimal boilerplate and immediately
  benchmark it against existing solutions.
- Create and maintain ground-truth annotations using the included tagging tool.
- Make data-driven decisions about OCR configuration rather than relying on
  manual inspection.

## Features

- **Unified OCR interface** -- integrate any OCR engine by subclassing
  `OCRAbstract`. Ships with EasyOCR and PaddleOCR implementations.
- **Robust preprocessing** -- a modular preprocessing layer that supports a
  wide range of composable operations, both built-in (grid splitting,
  binarization, contour-based segmentation) and user-supplied. Combine stages
  to construct the pipeline that fits your input data.
- **Evaluation pipeline** -- compute CER, WER, word recall, word precision,
  and word count ratio against ground-truth tags, with per-image and aggregate
  results including confidence intervals.
- **Visualization** -- bar charts, radar plots, and range plots for
  side-by-side comparison of OCR modules.
- **Tagging tool** -- browser-based UI for creating and editing ground-truth
  bounding-box annotations.

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

Execute all registered OCR modules against the dataset and generate metrics
and plots:

```bash
python -m scripts.run_evaluation [--config config.json] [--dataset dataset] [--output-dir scripts/results]
```

Results are saved to the output directory as JSON files and PNG plots.

### Draw Bounding Boxes

Overlay persisted OCR results on an image:

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
    "model_name": "MyOCRModule",
    "model_params": {"languages": ["en"]},
    "preprocess_methods": [
      {"name": "contour_split_image"}
    ]
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

## License

MIT -- see [LICENSE](LICENSE).
