"""Tests that the OCR tagging tool pages load correctly and contain expected elements."""

import unittest
from html.parser import HTMLParser

from tests.app_tests import APP_DEPS_REASON, HAS_APP_DEPS

if HAS_APP_DEPS:
    from scripts.image_ocr_tagging_tool.app import create_app


class _TagCounter(HTMLParser):
    """Counts open/close tags and collects ids and unclosed tags.

    Args:
        html: Not passed to __init__; call feed() after construction.
    """

    def __init__(self):
        super().__init__()
        self.open_count = 0
        self.close_count = 0
        self.ids = set()
        self.tag_stack = []
        self.void_elements = frozenset(
            [
                "area",
                "base",
                "br",
                "col",
                "embed",
                "hr",
                "img",
                "input",
                "link",
                "meta",
                "source",
                "track",
                "wbr",
            ]
        )

    def handle_starttag(self, tag, attrs):
        """Record opening tags, ids, and push non-void tags onto the stack."""
        self.open_count += 1
        attr_dict = dict(attrs)
        if "id" in attr_dict:
            self.ids.add(attr_dict["id"])
        if tag not in self.void_elements:
            self.tag_stack.append(tag)

    def handle_endtag(self, tag):
        """Record closing tags and pop matching tag from the stack."""
        self.close_count += 1
        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()


@unittest.skipUnless(HAS_APP_DEPS, APP_DEPS_REASON)
class TestTaggingToolPageLoads(unittest.TestCase):
    """Verify the index page returns 200 and is well-formed HTML."""

    def setUp(self):
        """Create test client."""
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def _get_index(self):
        """Fetch the index page and return (response, decoded_html).

        Returns:
            Tuple of (response object, html string).
        """
        resp = self.client.get("/")
        html = resp.data.decode("utf-8")
        return resp, html

    def test_index_returns_200(self):
        resp, _ = self._get_index()
        self.assertEqual(resp.status_code, 200)

    def test_index_content_type_is_html(self):
        resp, _ = self._get_index()
        self.assertIn("text/html", resp.content_type)

    def test_index_has_doctype(self):
        _, html = self._get_index()
        self.assertTrue(
            html.strip().lower().startswith("<!doctype html"),
            "Page should start with <!DOCTYPE html>",
        )

    def test_html_tags_are_balanced(self):
        _, html = self._get_index()
        parser = _TagCounter()
        parser.feed(html)
        self.assertEqual(
            len(parser.tag_stack),
            0,
            f"Unclosed tags remain on stack: {parser.tag_stack}",
        )


@unittest.skipUnless(HAS_APP_DEPS, APP_DEPS_REASON)
class TestTaggingToolUIElements(unittest.TestCase):
    """Verify all expected UI elements are present in the page."""

    def setUp(self):
        """Create test client and fetch page HTML."""
        app = create_app()
        app.config["TESTING"] = True
        self.client = app.test_client()
        resp = self.client.get("/")
        self.html = resp.data.decode("utf-8")

    def _assert_id_present(self, element_id):
        """Assert that an element with the given id exists in the HTML.

        Args:
            element_id: The DOM id attribute value to search for.
        """
        self.assertIn(
            f'id="{element_id}"',
            self.html,
            f"Expected element with id='{element_id}' not found",
        )

    def test_canvas(self):
        self._assert_id_present("canvas")

    def test_image_input(self):
        self._assert_id_present("imageInput")

    def test_image_input_allows_multiple(self):
        self.assertIn("multiple", self.html)

    def test_filename_display(self):
        self._assert_id_present("filenameDisplay")

    def test_json_input(self):
        self._assert_id_present("jsonInput")

    def test_tags_input(self):
        self._assert_id_present("tagsInput")

    def test_output_path(self):
        self._assert_id_present("outputPath")

    def test_browse_button(self):
        self.assertIn("browseOutputDir()", self.html)

    def test_ocr_model_select(self):
        self._assert_id_present("ocrModelSelect")

    def test_bb_list(self):
        self._assert_id_present("bbList")

    def test_status_msg(self):
        self._assert_id_present("statusMsg")

    def test_thumbs_container(self):
        self._assert_id_present("batchThumbs")

    def test_nav_section(self):
        self._assert_id_present("navSection")

    def test_nav_counter(self):
        self._assert_id_present("navCounter")

    def test_nav_starts_hidden(self):
        idx = self.html.find('id="navSection"')
        snippet = self.html[max(0, idx - 100) : idx]
        self.assertIn("hidden", snippet)


@unittest.skipUnless(HAS_APP_DEPS, APP_DEPS_REASON)
class TestTaggingToolFooterButtons(unittest.TestCase):
    """Verify footer buttons call the correct JS functions."""

    def setUp(self):
        """Create test client and fetch page HTML."""
        app = create_app()
        app.config["TESTING"] = True
        self.client = app.test_client()
        resp = self.client.get("/")
        self.html = resp.data.decode("utf-8")

    def test_save_current_button(self):
        self.assertIn("saveCurrent()", self.html)

    def test_save_all_button(self):
        self.assertIn("saveAll()", self.html)

    def test_clear_all_button(self):
        self.assertIn("clearAll()", self.html)

    def test_prev_button(self):
        self.assertIn("navPrev()", self.html)

    def test_next_button(self):
        self.assertIn("navNext()", self.html)


@unittest.skipUnless(HAS_APP_DEPS, APP_DEPS_REASON)
class TestTaggingToolJavaScriptFunctions(unittest.TestCase):
    """Verify critical JS functions are defined in the page script."""

    def setUp(self):
        """Create test client and fetch page HTML."""
        app = create_app()
        app.config["TESTING"] = True
        self.client = app.test_client()
        resp = self.client.get("/")
        self.html = resp.data.decode("utf-8")

    REQUIRED_FUNCTIONS = [
        "saveState",
        "loadImage",
        "updateNavUI",
        "navPrev",
        "navNext",
        "saveCurrent",
        "saveAll",
        "renderThumbs",
        "browseOutputDir",
        "clearAll",
        "redraw",
        "renderBBList",
        "runOCR",
        "runOCROnEmpty",
        "runOCRWholeImage",
        "loadOCRModels",
        "resizeCanvas",
    ]

    def test_all_required_functions_defined(self):
        for fn_name in self.REQUIRED_FUNCTIONS:
            with self.subTest(function=fn_name):
                self.assertIn(
                    f"function {fn_name}(",
                    self.html,
                    f"JS function '{fn_name}' not found in page",
                )


@unittest.skipUnless(HAS_APP_DEPS, APP_DEPS_REASON)
class TestTaggingToolJavaScriptVariables(unittest.TestCase):
    """Verify critical JS state variables are declared in the page."""

    def setUp(self):
        """Create test client and fetch page HTML."""
        app = create_app()
        app.config["TESTING"] = True
        self.client = app.test_client()
        resp = self.client.get("/")
        self.html = resp.data.decode("utf-8")

    REQUIRED_VARIABLES = [
        "loadedImages",
        "imageIndex",
        "originalImage",
        "boundingBoxes",
        "currentFilename",
    ]

    def test_all_required_variables_declared(self):
        for var_name in self.REQUIRED_VARIABLES:
            with self.subTest(variable=var_name):
                self.assertIn(
                    var_name,
                    self.html,
                    f"JS variable '{var_name}' not found in page",
                )
