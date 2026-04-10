from dataclasses import dataclass

import numpy as np


@dataclass
class InputImage:
    image: np.ndarray
    x_offset: int = 0
    y_offset: int = 0
