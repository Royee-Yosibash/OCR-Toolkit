from dataclasses import dataclass

import numpy as np


@dataclass
class InputImage:
    """An image array paired with its position relative to the original source image.

    Preprocessing operations (cropping, splitting) produce new ``InputImage``
    instances whose offsets accumulate, so detections made on a sub-image can
    later be remapped to coordinates in the original full image.

    Attributes:
        image: The image pixel data as a NumPy array.
        x_offset: Horizontal offset (in pixels) of this image's top-left
            corner relative to the original source image.
        y_offset: Vertical offset (in pixels) of this image's top-left
            corner relative to the original source image.
    """

    image: np.ndarray
    x_offset: int = 0
    y_offset: int = 0
