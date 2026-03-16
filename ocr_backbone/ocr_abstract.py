import copy
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_result import OCRResult


class OCRAbstract(ABC):
    """Abstract base class for OCR engines.

    Handles splitting an image into a grid of sub-images, running OCR on each
    cell, and remapping the resulting bounding boxes back to the original
    image coordinates.

    Subclasses must implement ``__init__`` to set up the underlying engine
    and ``_run_single`` to perform inference on a single sub-image.

    Subclasses are auto-registered by class name for lookup via ``from_config``.
    """

    _registry: dict = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Auto-register concrete subclasses by class name."""
        super().__init_subclass__(**kwargs)
        OCRAbstract._registry[cls.__name__] = cls

    @classmethod
    def from_config(cls, config: Union[OCRConfig, dict]):
        """Create an OCR instance from a config.

        Looks up the registered subclass matching ``config.model_name``
        and instantiates it, storing the config for later use.

        Args:
            config: The OCR run configuration.

        Returns:
            An instance of the matching OCR subclass.

        Raises:
            ValueError: If no subclass is registered for the model name.
        """
    
        model_name = config.model_name if isinstance(config, OCRConfig) else config["model_name"]
        if model_name not in cls._registry:
            raise ValueError(f"Unknown model: {model_name}")
        instance = cls._registry[model_name](config=config)
        return instance

    def __init__(self, config: Union[OCRConfig, dict]) -> None:
        """Initialize the OCR engine.
        """
        self.config = config if isinstance(config, OCRConfig) else OCRConfig.from_dict(config) 

    @abstractmethod
    def _run_single(self, image: np.ndarray, single_run_model_params: dict) -> OCRResult:
        """Run OCR on a single image.

        Args:
            image: Input image as a numpy array (H x W x C).
            single_run_model_params: Model-specific runtime parameters for this run.

        Returns:
            An OCRResult containing detected text regions.
        """

    def _split_image(
        self, image: np.ndarray, rows: int, cols: int
    ) -> list[list[np.ndarray]]:
        """Split an image into a grid of sub-images.

        Args:
            image: Input image as a numpy array (H x W x C).
            rows: Number of rows in the grid.
            cols: Number of columns in the grid.

        Returns:
            A 2D list (rows x cols) of sub-image numpy arrays.
        """
        h, w = image.shape[:2]
        row_edges = np.linspace(0, h, rows + 1, dtype=int)
        col_edges = np.linspace(0, w, cols + 1, dtype=int)
        grid = []
        for r in range(rows):
            row_cells = []
            for c in range(cols):
                cell = image[row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]]
                row_cells.append(cell)
            grid.append(row_cells)
        return grid

    def get_text_bb(self, image: np.ndarray, config_overrides: dict | None = None) -> OCRResult:
        """Run OCR over a grid of sub-images and return all detected text regions.

        Splits the image into a grid defined by the stored config, runs
        ``_run_single`` on each cell, and remaps the bounding boxes back to
        the original image coordinate space.

        Any keys in ``config_overrides`` are merged into ``model_params``
        for this run only; the stored config is not modified.

        Args:
            image: Input image as a numpy array (H x W x C).
            config_overrides: Optional dict of model_params overrides
                applied only for this run.

        Returns:
            An OCRResult with bounding boxes in original image coordinates.
        """
        if config_overrides:
            config = copy.deepcopy(self.config)
            config.update(config_overrides)
        else:
            config = self.config

        rows, cols = config.grid_rows, config.grid_cols
        h, w = image.shape[:2]
        row_edges = np.linspace(0, h, rows + 1, dtype=int)
        col_edges = np.linspace(0, w, cols + 1, dtype=int)

        grid = self._split_image(image, rows, cols)
        all_bboxes: list[BoundingBox] = []

        for r in range(rows):
            for c in range(cols):
                cell = grid[r][c]
                cell_result = self._run_single(cell, config.model_params)
                x_offset = int(col_edges[c])
                y_offset = int(row_edges[r])
                for bb in cell_result.bounding_boxes:
                    bb._remap_bounding_box(x_offset, y_offset)
                all_bboxes += cell_result.bounding_boxes

        if config.bb_validator is not None:
            all_bboxes = [bb for bb in all_bboxes if config.bb_validator(bb)]

        return OCRResult(bounding_boxes=all_bboxes)
