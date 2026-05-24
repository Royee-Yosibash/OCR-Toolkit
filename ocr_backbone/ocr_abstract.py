import copy
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

import numpy as np

from ocr_backbone.input_image import InputImage
from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_result import OCRResult
from ocr_backbone.polygon import Polygon
from utils.serialize_utils import register_unique

logger = logging.getLogger(__name__)


class OCRAbstract(ABC):
    """Abstract base class for OCR engines.

    Handles splitting an image into sub-images, running OCR on each cell,
    and remapping the resulting detections back to the original image
    coordinates.

    Subclasses must implement ``__init__`` to set up the underlying engine
    and ``_run_single`` to perform inference on a single sub-image.

    Subclasses are auto-registered by class name for lookup via ``from_config``.
    """

    _registry: dict[str, type] = {}

    #: ``model_params`` keys consumed at construction time and therefore
    #: stripped from per-call kwargs before being forwarded to
    #: ``_run_single``. Subclasses override this to declare engine-init
    #: parameters that must not leak into the inference call.
    _INIT_PARAM_KEYS: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs):
        """Register every concrete OCR subclass by class name."""
        super().__init_subclass__(**kwargs)
        register_unique(OCRAbstract._registry, cls)

    @classmethod
    def registered_models(cls) -> list[str]:
        """Return the names of every registered OCR subclass.

        Returns:
            A list of class names available via ``from_config``.
        """
        return list(cls._registry.keys())

    @classmethod
    def from_config(cls, config: OCRConfig | dict):
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
        logger.info("Initialized module %s", model_name)
        return instance

    def __init__(self, config: OCRConfig | dict, alias: str = "") -> None:
        """Initialize the OCR engine.

        Args:
            config: The OCR run configuration.
            alias: Optional display name used as the label in evaluations.
                When empty, the default ``{ClassName}_{i}`` label is used.
        """
        self.config = config if isinstance(config, OCRConfig) else OCRConfig.from_dict(config)
        self._alias = alias

    @property
    def alias(self) -> str:
        """Read-only display name for this OCR engine.

        Returns:
            The alias string, or empty string if none was set.
        """
        return self._alias

    @abstractmethod
    def _run_single(self, image: np.ndarray, single_run_model_params: dict) -> OCRResult:
        """Run OCR on a single image.

        Args:
            image: Input image as a numpy array (H x W x C).
            single_run_model_params: Model-specific runtime parameters for this run.

        Returns:
            An OCRResult containing detected text regions.

        """

    @staticmethod
    def _pp_method_name(method: Callable) -> str:
        """Return a human-readable name for a preprocessing callable.

        Args:
            method: The preprocessing callable.

        Returns:
            A string such as ``"binarize"`` or ``"binarize(method='otsu')"``.
        """
        if isinstance(method, partial):
            name = method.func.__qualname__
            if method.keywords:
                args = ", ".join(f"{k}={v!r}" for k, v in method.keywords.items())
                return f"{name}({args})"
            return name
        return method.__qualname__

    def _preprocess(self, image: np.ndarray, config: OCRConfig) -> list[InputImage]:
        """Run the preprocessing pipeline defined in the config.

        Args:
            image: Input image as a numpy array (H x W x C).
            config: The OCR config containing preprocessing callables.

        Returns:
            A list of InputImage cells ready for OCR inference.
        """
        step_names = [self._pp_method_name(m) for m in config.preprocess_methods]
        logger.info(
            "Preprocessing pipeline: %s",
            " -> ".join(step_names) if step_names else "(none)",
        )

        input_images = [InputImage(image=image)]

        for idx, method in enumerate(config.preprocess_methods):
            logger.info(
                "Running preprocessing step %d/%d: %s (%d input image(s))",
                idx + 1,
                len(config.preprocess_methods),
                step_names[idx],
                len(input_images),
            )

            outputs = []
            for in_img in input_images:
                res = method(in_img)
                outputs += res if isinstance(res, list) else [res]

            input_images = outputs
            logger.debug(
                "Step %d/%d produced %d image(s)",
                idx + 1,
                len(config.preprocess_methods),
                len(input_images),
            )

        logger.info("Preprocessing complete: %d image(s) to process", len(input_images))
        return input_images

    def get_text_detections(self, image: np.ndarray, config_overrides: dict | None = None) -> OCRResult:
        """Run OCR over a grid of sub-images and return all detected text regions.

        Splits the image into a grid defined by the stored config, runs
        ``_run_single`` on each cell, and remaps the detections back to
        the original image coordinate space.

        Any keys in ``config_overrides`` are merged into ``model_params``
        for this run only; the stored config is not modified.

        Args:
            image: Input image as a numpy array (H x W x C).
            config_overrides: Optional dict of model_params overrides
                applied only for this run.

        Returns:
            An OCRResult with detections in original image coordinates.
        """
        logger.info("get_text_detections called with image shape %s", image.shape)
        if config_overrides:
            logger.info("Applying config overrides: %s", list(config_overrides.keys()))
            config = copy.deepcopy(self.config)
            config.update(config_overrides)
        else:
            config = self.config

        cells = self._preprocess(image=image, config=config)

        logger.info("Running OCR on %d cell(s)", len(cells))
        call_params = {k: v for k, v in config.model_params.items() if k not in self._INIT_PARAM_KEYS}
        all_detections: list[Polygon] = []
        for cell in cells:
            cell_result = self._run_single(cell.image, call_params)
            all_detections.extend(d.translate_coordinates(cell.x_offset, cell.y_offset) for d in cell_result.detections)

        if config.detection_validator is not None:
            before = len(all_detections)
            all_detections = [det for det in all_detections if config.detection_validator(det)]
            logger.info("detection_validator filtered %d -> %d detections", before, len(all_detections))

        logger.info("Returning %d detection(s)", len(all_detections))
        return OCRResult(detections=all_detections)
