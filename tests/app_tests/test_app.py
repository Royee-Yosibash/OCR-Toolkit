import base64
import json
import unittest
from unittest.mock import patch

from tests.app_tests import APP_DEPS_REASON, HAS_APP_DEPS
from tests.consts import TEST_IMAGE_PATH
from tests.unit_tests.dummy_ocr import DummyOCR  # noqa: F401 - register in OCRAbstract

if HAS_APP_DEPS:
    from scripts.image_ocr_tagging_tool.app import create_app


@unittest.skipUnless(HAS_APP_DEPS, APP_DEPS_REASON)
class TestSaveBatchEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def _make_bb(self, text="hello"):
        """Create a minimal bounding box dict.

        Args:
            text: The text content for the bounding box.

        Returns:
            A bounding box dict with coordinates, text, and confidence.
        """
        return {
            "_type": "BoundingBox",
            "coordinates": [[0, 0], [10, 10]],
            "text": text,
            "confidence": 0.9,
        }

    @patch("scripts.image_ocr_tagging_tool.app.save_json")
    def test_save_single_ok(self, mock_save):
        payload = {
            "detections": [self._make_bb("hello")],
            "tags": ["printed"],
            "filename": "img.png",
        }
        resp = self.client.post(
            "/api/save",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("img.json", body["path"])
        self.assertEqual(mock_save.call_count, 1)

    def test_save_single_missing_detections_returns_400(self):
        payload = {
            "bounding_boxes": [self._make_bb("hello")],
            "filename": "img.png",
        }
        resp = self.client.post(
            "/api/save",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("Missing required field", body["error"])

    def test_save_single_empty_text_returns_400(self):
        payload = {
            "detections": [self._make_bb("")],
            "filename": "img.png",
        }
        resp = self.client.post(
            "/api/save",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("no text", body["error"])

    @patch("scripts.image_ocr_tagging_tool.app.save_json")
    def test_save_batch_ok(self, mock_save):
        payload = {
            "images": [
                {
                    "detections": [self._make_bb("a")],
                    "tags": ["printed"],
                    "filename": "img1.png",
                },
                {
                    "detections": [self._make_bb("b")],
                    "tags": [],
                    "filename": "img2.png",
                },
            ],
        }
        resp = self.client.post(
            "/api/save_batch",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(len(body["paths"]), 2)
        self.assertEqual(mock_save.call_count, 2)

    def test_save_batch_empty_text_returns_400(self):
        payload = {
            "images": [
                {
                    "detections": [self._make_bb("ok")],
                    "filename": "good.png",
                },
                {
                    "detections": [self._make_bb("")],
                    "filename": "bad.png",
                },
            ],
        }
        resp = self.client.post(
            "/api/save_batch",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("bad.png", body["error"])
        self.assertIn("#1", body["error"])

    def test_save_batch_missing_images_returns_400(self):
        resp = self.client.post(
            "/api/save_batch",
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("Missing required field", body["error"])

    @patch("scripts.image_ocr_tagging_tool.app.save_json")
    def test_save_batch_with_output_path(self, mock_save):
        payload = {
            "images": [
                {
                    "detections": [self._make_bb("x")],
                    "filename": "pic.png",
                },
            ],
            "output_path": "/tmp/ocr_test_output/",
        }
        resp = self.client.post(
            "/api/save_batch",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn("pic.json", body["paths"][0])

    @patch("scripts.image_ocr_tagging_tool.app.save_json")
    def test_run_ocr_then_save(self, mock_save):
        image_b64 = base64.b64encode(TEST_IMAGE_PATH.read_bytes()).decode()

        ocr_resp = self.client.post(
            "/api/run_ocr",
            data=json.dumps({"image": image_b64, "model_name": "DummyOCR"}),
            content_type="application/json",
        )
        self.assertEqual(ocr_resp.status_code, 200)
        ocr_data = ocr_resp.get_json()
        self.assertIn("detections", ocr_data)
        self.assertGreater(len(ocr_data["detections"]), 0)

        save_payload = {
            "detections": ocr_data["detections"],
            "tags": ["test"],
            "filename": "pipeline.png",
        }
        save_resp = self.client.post(
            "/api/save",
            data=json.dumps(save_payload),
            content_type="application/json",
        )
        self.assertEqual(save_resp.status_code, 200)
        self.assertEqual(save_resp.get_json()["status"], "ok")
