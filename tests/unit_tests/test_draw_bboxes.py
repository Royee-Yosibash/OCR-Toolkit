import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.draw_bboxes import draw_bboxes


class TestDrawBboxes(unittest.TestCase):
    """Tests for the draw_bboxes script."""

    def test_draw_bboxes_creates_output(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image = np.zeros((100, 200, 3), dtype=np.uint8)
            image_path = tmp_path / "sample.png"
            cv2.imwrite(str(image_path), image)

            results = {
                "bounding_boxes": [
                    {"coordinates": [[10, 10], [50, 30]], "text": "hello", "confidence": 0.95},
                    {"coordinates": [[60, 50], [150, 80]], "text": "world", "confidence": 0.8},
                ]
            }
            results_path = tmp_path / "results.json"
            results_path.write_text(json.dumps(results))

            output = draw_bboxes(str(image_path), str(results_path))
            output_path = Path(output)

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.name, "sample_with_bounding_box.png")

            output_image = cv2.imread(str(output_path))
            self.assertIsNotNone(output_image)
            self.assertFalse(np.array_equal(output_image, image))
