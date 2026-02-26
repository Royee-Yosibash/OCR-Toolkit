"""Draw bounding boxes on an image from a saved OCR results JSON.

Usage:
    python -m scripts.draw_bboxes <image_path> <results_json>

Creates a new image file named <image_stem>_with_bounding_box.<ext> in the
same directory as the original image.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from ocr_backbone.ocr_result import OCRResult
from utils.json_utils import load_json

BOX_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.25
FONT_THICKNESS = 1
LABEL_GAP = 3


def draw_bboxes(image_path: str, results_path: str) -> str:
    """Draw bounding boxes on an image and save the result.

    Args:
        image_path: Path to the input image.
        results_path: Path to the JSON file with saved OCR results.

    Returns:
        The path to the output image.
    """
    image = cv2.imread(image_path)
    ocr_result = OCRResult.from_dict(load_json(Path(results_path)))
    bboxes = ocr_result.bounding_boxes

    for i, bb in enumerate(bboxes):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        top_left, bottom_right = bb.coordinates
        cv2.rectangle(image, top_left, bottom_right, color, BOX_THICKNESS)

        label = f"{bb.text} ({bb.confidence:.2f})"
        text_size, baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        text_x = bottom_right[0] - text_size[0]
        text_y = bottom_right[1] + LABEL_GAP + text_size[1]
        text_x = max(text_x, 0)
        text_y = min(text_y, image.shape[0] - 1)

        cv2.putText(
            image, label, (text_x, text_y),
            FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA,
        )

    img_path = Path(image_path)
    output_path = img_path.parent / f"{img_path.stem}_with_bounding_box{img_path.suffix}"
    cv2.imwrite(str(output_path), image)
    return str(output_path)


def main() -> None:
    """Parse arguments and draw bounding boxes."""
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes on an image from saved OCR results."
    )
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("results_json", help="Path to the saved OCR results JSON.")
    args = parser.parse_args()

    output = draw_bboxes(args.image_path, args.results_json)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
