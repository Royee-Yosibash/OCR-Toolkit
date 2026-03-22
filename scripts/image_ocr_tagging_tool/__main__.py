"""Entry point for the OCR tagging tool backend.

Imports all OCR modules to trigger subclass registration, creates the
Flask application, and opens the UI in the default browser.

Usage::

    python -m scripts.image_ocr_tagging_tool [--port PORT]

Args:
    --port PORT: Port to run the Flask server on (default: 5000).

The server starts on http://localhost:PORT and the browser opens
automatically.
"""

import argparse
import webbrowser

from ocr_modules import import_all_modules


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

    import_all_modules()

    from scripts.image_ocr_tagging_tool.app import create_app

    app = create_app()

    webbrowser.open(f"http://localhost:{args.port}")
    app.run(host="localhost", port=args.port)


if __name__ == "__main__":
    main()
