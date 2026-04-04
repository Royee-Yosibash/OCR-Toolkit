import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.image_ocr_tagging_tool.app import create_app


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
            "coordinates": [[0, 0], [10, 10]],
            "text": text,
            "confidence": 0.9,
        }

    @patch("scripts.image_ocr_tagging_tool.app.save_json")
    def test_save_batch_ok(self, mock_save):
        payload = {
            "images": [
                {
                    "bounding_boxes": [self._make_bb("a")],
                    "tags": ["printed"],
                    "filename": "img1.png",
                },
                {
                    "bounding_boxes": [self._make_bb("b")],
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
                    "bounding_boxes": [self._make_bb("ok")],
                    "filename": "good.png",
                },
                {
                    "bounding_boxes": [self._make_bb("")],
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
                    "bounding_boxes": [self._make_bb("x")],
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
