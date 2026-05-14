"""End-to-end tests for the OCR tagging tool using Playwright.

Spawns the real Flask app in a background thread and drives the page in
a headless browser, exercising the JS functions ``runOCRWholeImage`` and
``runOCROnEmpty`` against the live backend.

These tests catch regressions that pure HTML/string-matching tests
cannot, e.g. the frontend reading a wrong response field name.
"""

import socket
import threading
import time
import unittest

from tests.app_tests import (
    APP_DEPS_REASON,
    HAS_APP_DEPS,
    HAS_PLAYWRIGHT,
    PLAYWRIGHT_REASON,
)
from tests.consts import TEST_IMAGE_PATH

if HAS_APP_DEPS:
    from werkzeug.serving import make_server

    from scripts.image_ocr_tagging_tool.app import create_app

if HAS_PLAYWRIGHT:
    from playwright.sync_api import sync_playwright


def _free_port() -> int:
    """Pick a free TCP port from the OS.

    Returns:
        A currently-unbound port number.
    """
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _ServerThread(threading.Thread):
    """Run a werkzeug server inside a daemon thread."""

    def __init__(self, app, host: str, port: int) -> None:
        """Set up the server but don't start it.

        Args:
            app: A Flask app instance.
            host: Host to bind to.
            port: Port to bind to.
        """
        super().__init__(daemon=True)
        self.srv = make_server(host, port, app)

    def run(self) -> None:
        """Serve until shutdown is called."""
        self.srv.serve_forever()

    def shutdown(self) -> None:
        """Stop the server."""
        self.srv.shutdown()


@unittest.skipUnless(HAS_APP_DEPS, APP_DEPS_REASON)
@unittest.skipUnless(HAS_PLAYWRIGHT, PLAYWRIGHT_REASON)
class TestTaggingToolE2E(unittest.TestCase):
    """End-to-end tests driving the live page in a headless browser."""

    @classmethod
    def setUpClass(cls) -> None:
        """Start a Flask server in a thread and a Chromium browser."""
        cls.app = create_app()
        cls.port = _free_port()
        cls.server = _ServerThread(cls.app, "127.0.0.1", cls.port)
        cls.server.start()

        cls.base_url = f"http://127.0.0.1:{cls.port}"
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", cls.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)

        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch()

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down browser and server."""
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()

    def setUp(self) -> None:
        """Open a fresh browser context and page for each test."""
        self.context = self.browser.new_context()
        self.page = self.context.new_page()

    def tearDown(self) -> None:
        """Close the browser context."""
        self.context.close()

    def _open_app_with_image(self) -> None:
        """Navigate to the app, select DummyOCR, and load the test image."""
        self.page.goto(self.base_url)
        self.page.wait_for_function("document.querySelector('#ocrModelSelect option[value=\"DummyOCR\"]') !== null")
        self.page.select_option("#ocrModelSelect", "DummyOCR")
        self.page.set_input_files("#imageInput", str(TEST_IMAGE_PATH))
        self.page.wait_for_function("originalImage !== null && originalImage.complete")

    def test_run_ocr_whole_image_populates_bbs(self):
        """OCR-Whole-Image button populates the BB list with detections."""
        self._open_app_with_image()

        self.page.click("button[onclick='runOCRWholeImage()']")
        self.page.wait_for_function("boundingBoxes.length >= 1", timeout=5000)

        texts = self.page.evaluate("boundingBoxes.map(b => b.text)")
        self.assertGreaterEqual(len(texts), 1)
        self.assertTrue(all(t.strip() != "" for t in texts), texts)

    def test_run_ocr_on_empty_fills_in_text(self):
        """Empty BBs are populated with text after clicking 'OCR Empty BBs'."""
        self._open_app_with_image()

        self.page.evaluate(
            """() => {
                boundingBoxes.push({ x1: 10, y1: 10, x2: 200, y2: 100, text: '' });
                renderBBList();
                redraw();
            }"""
        )

        self.page.click("button[onclick='runOCROnEmpty()']")
        self.page.wait_for_function(
            "boundingBoxes.length >= 1 && boundingBoxes[0].text.trim() !== ''",
            timeout=5000,
        )

        text = self.page.evaluate("boundingBoxes[0].text")
        self.assertNotEqual(text.strip(), "")
