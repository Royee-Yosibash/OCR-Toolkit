"""Entry point for the OCR tagging tool backend.

Imports all OCR modules to trigger subclass registration, creates the
Flask application, and opens the UI in the default browser.
"""

import argparse
import importlib
import pkgutil
import webbrowser
from pathlib import Path

import ocr_modules


def _import_all_modules() -> None:
    """Import every module inside the ocr_modules package to trigger
    subclass registration in OCRAbstact._registry.
    """
    package_path = Path(ocr_modules.__file__).parent
    for finder, name, is_pkg in pkgutil.iter_modules([str(package_path)]):
        importlib.import_module(f"ocr_modules.{name}")


def main() -> None:
    """Parse arguments, start the Flask server, and open the browser.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Start the OCR tagging tool backend.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the Flask server on (default: 5000).",
    )
    args = parser.parse_args()

    _import_all_modules()

    from scripts.image_ocr_tagging_tool.app import create_app

    app = create_app()

    webbrowser.open(f"http://localhost:{args.port}")
    app.run(host="localhost", port=args.port)


if __name__ == "__main__":
    """Run the OCR tagging tool backend.

    Usage::

        python -m scripts.tagging_images_ocr [--port PORT]

    Args:
        --port PORT: Port to run the Flask server on (default: 5000).

    The server starts on http://localhost:PORT and the browser opens
    automatically.
    """
    main()
