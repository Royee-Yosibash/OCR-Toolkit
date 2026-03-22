import numpy as np
from dataclasses import dataclass


@dataclass
class InputImage:
    image: np.ndarray
    x_offset: int = 0
    y_offset: int = 0