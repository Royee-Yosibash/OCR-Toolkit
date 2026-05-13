"""Generic image array transformations that operate on raw numpy images."""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 view of an image regardless of input shape.

    Args:
        image: Input image as a numpy array. May be 2D (already grayscale),
            3D with a single channel, or 3D with three BGR channels.

    Returns:
        A 2D single-channel image.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.squeeze() if image.ndim == 3 else image
