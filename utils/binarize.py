import cv2
import numpy as np


def otsu_binarize(image: np.ndarray) -> np.ndarray:
    """Binarize a grayscale image using Otsu's automatic thresholding.

    Args:
        image: Single-channel uint8 image.

    Returns:
        Binary uint8 image with values 0 or 255.
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
