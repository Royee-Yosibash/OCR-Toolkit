"""Entry point for the OCR tagging tool backend.

Creates the Flask application (which itself triggers OCR engine
registration) and opens the UI in the default browser.

Usage::

    python -m scripts.image_ocr_tagging_tool [--port PORT]

Args:
    --port PORT: Port to run the Flask server on (default: 5000).

The server starts on http://localhost:PORT and the browser opens
automatically.
"""

import argparse
import logging
import webbrowser

logger = logging.getLogger(__name__)


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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    from scripts.image_ocr_tagging_tool.app import create_app

    logger.info("Creating Flask app")
    app = create_app()

    logger.info("Opening browser at http://localhost:%d", args.port)
    webbrowser.open(f"http://localhost:{args.port}")

    logger.info("Starting Flask server on port %d", args.port)
    app.run(host="localhost", port=args.port)


if __name__ == "__main__":
    main()
