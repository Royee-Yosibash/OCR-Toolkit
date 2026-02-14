from abc import ABC, abstractmethod

import numpy as np

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.config import OCRConfig
from ocr_backbone.ocr_result import OCRResult


class OCRAbstact(ABC):
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
        OCRAbstact._registry[cls.__name__] = cls

    @classmethod
    def from_config(cls, config: OCRConfig):
        """Create an OCR instance from a config.

        Looks up the registered subclass matching ``config.model_name``
        and instantiates it.

        Args:
            config: The OCR run configuration.

        Returns:
            An instance of the matching OCR subclass.

        Raises:
            ValueError: If no subclass is registered for the model name.
        """
        ocr_cls = cls._registry.get(config.model_name)
        if ocr_cls is None:
            raise ValueError(
                f"Unknown model {config.model_name!r}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return ocr_cls()

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the OCR engine."""

    @abstractmethod
    def _run_single(self, image: np.ndarray, module_config: OCRConfig) -> OCRResult:
        """Run OCR on a single image.

        Args:
            image: Input image as a numpy array (H x W x C).
            module_config: The OCR run configuration.

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

    def get_text_bb(self, image: np.ndarray, config: OCRConfig) -> OCRResult:
        """Run OCR over a grid of sub-images and return all detected text regions.

        Splits the image into a grid defined by ``config.grid_rows`` and
        ``config.grid_cols``, runs ``_run_single`` on each cell, and remaps
        the bounding boxes back to the original image coordinate space.

        Args:
            image: Input image as a numpy array (H x W x C).
            config: The OCR run configuration.

        Returns:
            An OCRResult with bounding boxes in original image coordinates.
        """
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
