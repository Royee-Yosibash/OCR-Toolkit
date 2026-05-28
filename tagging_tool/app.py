"""Flask application factory for the OCR tagging tool."""

import base64
import io
import logging
import subprocess
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.exceptions import HTTPException

from evaluation.ocr_ground_truth import OCRGroundTruth
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_modules import import_all_modules
from utils.json_utils import save_json

logger = logging.getLogger(__name__)


def _resolve_save_path(output_path: str, filename: str) -> Path:
    """Compute the destination JSON path for a single tagged image.

    Pure path computation -- no filesystem side effects. Parent
    directories are created by the writer (``save_json(..., mkdir=True)``).

    Args:
        output_path: Caller-supplied path. May be empty (use default), a
            directory, or an explicit file path.
        filename: Original image filename, used to derive the JSON stem.

    Returns:
        The resolved file path.
    """
    stem = Path(filename).stem
    if not output_path:
        return Path.home() / "Downloads" / "ocr_tags" / f"{stem}.json"
    save_path = Path(output_path)
    if save_path.is_dir() or not save_path.suffix:
        save_path = save_path / f"{stem}.json"
    return save_path


def _validate_no_empty_text(detections: list[dict], context: str = "") -> None:
    """Reject any detection whose text is empty or whitespace.

    Args:
        detections: List of detection dicts with a "text" key.
        context: Optional prefix (e.g. ``"Image 'foo.png' (index 0)"``)
            inserted at the start of the error message for batch callers.

    Raises:
        ValueError: If at least one detection has empty/whitespace text.
            The message lists the 1-based indices of the offenders.
    """
    empty = [i + 1 for i, det in enumerate(detections) if not det.get("text", "").strip()]
    if not empty:
        return
    indices = ", ".join(str(i) for i in empty)
    prefix = f"{context}: " if context else ""
    raise ValueError(f"{prefix}Detection(s) #{indices} have no text. Fill in or delete them before saving.")


def _save_one(detections: list[dict], tags: list[str], filename: str, output_path: str) -> Path:
    """Persist a single image's tagged detections.

    Args:
        detections: List of detection dicts.
        tags: List of tag strings.
        filename: Original image filename, used to derive the JSON stem.
        output_path: Caller-supplied output path; see ``_resolve_save_path``.

    Returns:
        The path where the JSON was written.
    """
    save_path = _resolve_save_path(output_path, filename)
    result = OCRGroundTruth.from_dict({"detections": detections, "tags": tags})
    save_json(save_path, result.to_dict(), mkdir=True)
    return save_path


def create_app() -> Flask:
    """Create and configure the Flask application.

    Triggers OCR engine registration so that ``OCRAbstract.registered_models()``
    is populated regardless of how the app is launched (CLI, tests, WSGI).
    Safe to call multiple times because module imports are cached.

    Returns:
        A Flask app instance with all routes registered.
    """
    import_all_modules()
    app = Flask(__name__)

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception):
        """Return any unhandled exception as a JSON 500 so API clients get a parseable body.

        HTTP exceptions (e.g. 404 from missing routes like ``/favicon.ico``)
        are passed through to Flask's default handling so they don't get
        logged as unhandled errors or rewrapped as 500s.

        Args:
            exc: The exception raised by a route handler.

        Returns:
            A JSON response with the exception message and HTTP 500 status,
            or the original HTTPException for Flask to render normally.
        """
        if isinstance(exc, HTTPException):
            return exc
        logger.exception("Unhandled error in %s", request.path)
        return jsonify({"error": str(exc)}), 500

    @app.route("/")
    def index():
        """Render the main tagging UI page.

        Returns:
            Rendered index.html template.
        """
        return render_template("index.html")

    @app.route("/api/ocr_models", methods=["GET"])
    def ocr_models():
        """Return a JSON list of registered OCR model names.

        Returns:
            JSON response with a list of model name strings.
        """
        return jsonify(OCRAbstract.registered_models())

    @app.route("/api/run_ocr", methods=["POST"])
    def run_ocr():
        """Run OCR on a base64-encoded image crop.

        Expects a JSON body with:
            image: Base64-encoded image data.
            model_name: Name of the registered OCR model to use.

        Returns:
            JSON response with the OCR result dict.
        """
        try:
            logger.info("/api/run_ocr called")
            data = request.get_json()
            image_b64 = data["image"]
            model_name = data["model_name"]
            logger.info("Request payload: model_name=%s, image_b64_length=%d", model_name, len(image_b64))

            logger.info("Decoding base64 image")
            image_bytes = base64.b64decode(image_b64)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_array = np.array(pil_image)
            logger.info("Decoded image to array with shape %s", image_array.shape)

            config = {"model_name": model_name, "grid_rows": 1, "grid_cols": 1}
            logger.info("Instantiating OCR module from config: %s", config)
            ocr = OCRAbstract.from_config(config)

            logger.info("Running OCR with model %s", model_name)
            result = ocr.get_text_detections(image_array)
            logger.info("OCR complete: %d detection(s)", len(result.detections))

            return jsonify(result.to_dict())
        except KeyError as e:
            logger.warning("/api/run_ocr missing required field: %s", e)
            return jsonify({"error": f"Missing required field: {e}"}), 400
        except ValueError as e:
            logger.warning("/api/run_ocr value error: %s", e)
            return jsonify({"error": str(e)}), 400

    @app.route("/api/save", methods=["POST"])
    def save():
        """Save tagged bounding boxes to a JSON file.

        Expects a JSON body with:
            detections: List of detection dicts.
            tags: Optional list of tag strings.
            filename: Original image filename (used to derive output name).
            output_path: Optional path to save the JSON file. Defaults to
                ~/Downloads/ocr_tags/{filename_stem}.json.

        Returns:
            JSON response with status and the actual save path.
        """
        try:
            data = request.get_json()
            detections = data["detections"]
            tags = data.get("tags", [])
            filename = data.get("filename", "untitled.png")
            output_path = data.get("output_path", "")

            _validate_no_empty_text(detections)
            save_path = _save_one(detections, tags, filename, output_path)
            return jsonify({"status": "ok", "path": str(save_path)})
        except KeyError as e:
            return jsonify({"error": f"Missing required field: {e}"}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/browse_directory", methods=["POST"])
    def browse_directory():
        """Open a native OS directory picker dialog and return the selected path.

        Uses zenity on Linux. Falls back to an error message if no supported
        dialog tool is found.

        Returns:
            JSON response with the selected directory path, or an empty
            string if the user cancelled.
        """
        try:
            result = subprocess.run(
                ["zenity", "--file-selection", "--directory", "--title=Select Output Directory"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            chosen = result.stdout.strip() if result.returncode == 0 else ""
            return jsonify({"path": chosen})
        except FileNotFoundError:
            return jsonify({"error": "No directory picker available. Install zenity or type the path manually."}), 500

    @app.route("/api/save_batch", methods=["POST"])
    def save_batch():
        """Save tagged bounding boxes for multiple images at once.

        Expects a JSON body with:
            images: List of dicts, each containing:
                detections: List of detection dicts.
                tags: Optional list of tag strings.
                filename: Original image filename (used to derive output name).
            output_path: Optional global path to save the JSON files.
                Defaults to ~/Downloads/ocr_tags/{filename_stem}.json.

        Returns:
            JSON response with status and a list of saved paths.
        """
        try:
            data = request.get_json()
            images = data["images"]
            output_path = data.get("output_path", "")

            for img_idx, image_entry in enumerate(images):
                filename = image_entry.get("filename", "untitled.png")
                _validate_no_empty_text(
                    image_entry["detections"],
                    context=f"Image '{filename}' (index {img_idx})",
                )

            paths = [
                str(
                    _save_one(
                        image_entry["detections"],
                        image_entry.get("tags", []),
                        image_entry.get("filename", "untitled.png"),
                        output_path,
                    )
                )
                for image_entry in images
            ]

            return jsonify({"status": "ok", "paths": paths})
        except KeyError as e:
            return jsonify({"error": f"Missing required field: {e}"}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    return app
