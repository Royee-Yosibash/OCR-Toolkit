"""Generic image array transformations that operate on raw numpy images."""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 view of an image regardless of input shape.

    Three-channel inputs are assumed to be in RGB order, which is the project's
    convention (images are loaded via ``PIL.Image.open(...).convert("RGB")``).

    Args:
        image: Input image as a numpy array. May be 2D (already grayscale),
            3D with a single channel, or 3D with three RGB channels.

    Returns:
        A 2D single-channel image.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.squeeze() if image.ndim == 3 else image


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Return a 3-channel image regardless of input shape.

    Grayscale (2D) inputs are broadcast across three channels. RGBA inputs
    have their alpha channel dropped. RGB inputs are returned unchanged.

    Args:
        image: Input image as a numpy array. Accepted shapes are
            (H, W), (H, W, 3) or (H, W, 4).

    Returns:
        A 3D image with shape (H, W, 3).

    Raises:
        ValueError: If the input has an unsupported number of dimensions
            or channel count.
    """
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    if image.ndim == 3:
        if image.shape[2] == 3:
            return image
        if image.shape[2] == 4:
            return image[:, :, :3]
    raise ValueError(f"Unsupported image shape for RGB conversion: {image.shape}")
