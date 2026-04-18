"""Flask application factory for the OCR tagging tool."""

import base64
import io
import subprocess
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

from evaluation.ocr_ground_truth import OCRGroundTruth
from ocr_backbone.ocr_abstract import OCRAbstract
from utils.json_utils import save_json


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns:
        A Flask app instance with all routes registered.
    """
    app = Flask(__name__)

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
        return jsonify(list(OCRAbstract._registry.keys()))

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
            data = request.get_json()
            image_b64 = data["image"]
            model_name = data["model_name"]

            image_bytes = base64.b64decode(image_b64)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_array = np.array(pil_image)

            config = {"model_name": model_name, "grid_rows": 1, "grid_cols": 1}
            ocr = OCRAbstract.from_config(config)
            result = ocr.get_text_bb(image_array)

            return jsonify(result.to_dict())
        except KeyError as e:
            return jsonify({"error": f"Missing required field: {e}"}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/save", methods=["POST"])
    def save():
        """Save tagged bounding boxes to a JSON file.

        Expects a JSON body with:
            bounding_boxes: List of bounding box dicts.
            tags: Optional list of tag strings.
            filename: Original image filename (used to derive output name).
            output_path: Optional path to save the JSON file. Defaults to
                ~/Downloads/ocr_tags/{filename_stem}.json.

        Returns:
            JSON response with status and the actual save path.
        """
        try:
            data = request.get_json()
            bounding_boxes = data["bounding_boxes"]
            tags = data.get("tags", [])
            filename = data.get("filename", "untitled.png")
            output_path = data.get("output_path", "")

            empty_bbs = [i for i, bb in enumerate(bounding_boxes) if not bb.get("text", "").strip()]
            if empty_bbs:
                indices = ", ".join(str(i + 1) for i in empty_bbs)
                return jsonify({"error": f"BB(s) #{indices} have no text. Fill in or delete them before saving."}), 400

            if not output_path:
                filename_stem = Path(filename).stem
                save_dir = Path.home() / "Downloads" / "ocr_tags"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{filename_stem}.json"
            else:
                save_path = Path(output_path)
                if save_path.is_dir() or not save_path.suffix:
                    filename_stem = Path(filename).stem
                    save_path = save_path / f"{filename_stem}.json"
                save_path.parent.mkdir(parents=True, exist_ok=True)

            result = OCRGroundTruth.from_dict({"bounding_boxes": bounding_boxes, "tags": tags})
            save_json(save_path, result.to_dict())

            return jsonify({"status": "ok", "path": str(save_path)})
        except KeyError as e:
            return jsonify({"error": f"Missing required field: {e}"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

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
                bounding_boxes: List of bounding box dicts.
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
                bounding_boxes = image_entry["bounding_boxes"]
                empty_bbs = [i for i, bb in enumerate(bounding_boxes) if not bb.get("text", "").strip()]
                if empty_bbs:
                    indices = ", ".join(str(i + 1) for i in empty_bbs)
                    filename = image_entry.get("filename", "untitled.png")
                    return jsonify(
                        {
                            "error": f"Image '{filename}' (index {img_idx}): "
                            f"BB(s) #{indices} have no text. "
                            f"Fill in or delete them before saving."
                        }
                    ), 400

            paths = []
            for image_entry in images:
                bounding_boxes = image_entry["bounding_boxes"]
                tags = image_entry.get("tags", [])
                filename = image_entry.get("filename", "untitled.png")

                if not output_path:
                    filename_stem = Path(filename).stem
                    save_dir = Path.home() / "Downloads" / "ocr_tags"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"{filename_stem}.json"
                else:
                    save_path = Path(output_path)
                    if save_path.is_dir() or not save_path.suffix:
                        filename_stem = Path(filename).stem
                        save_path = save_path / f"{filename_stem}.json"
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                result = OCRGroundTruth.from_dict({"bounding_boxes": bounding_boxes, "tags": tags})
                save_json(save_path, result.to_dict())
                paths.append(str(save_path))

            return jsonify({"status": "ok", "paths": paths})
        except KeyError as e:
            return jsonify({"error": f"Missing required field: {e}"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app
