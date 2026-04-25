"""OCR module using the PaddleOCR engine."""

import logging

import numpy as np

from ocr_backbone.bounding_box import BoundingBox
from ocr_backbone.ocr_abstract import OCRAbstract
from ocr_backbone.ocr_config import OCRConfig
from ocr_backbone.ocr_result import OCRResult

logger = logging.getLogger(__name__)

try:
    from paddleocr import PaddleOCR

    _HAS_PADDLEOCR = True
except ImportError:
    _HAS_PADDLEOCR = False
    logger.info("paddleocr not installed -- PaddleOCRModule will not be available.")

MOBILE_DET = "PP-OCRv5_mobile_det"
MOBILE_REC = "PP-OCRv5_mobile_rec"
SERVER_DET = "PP-OCRv5_server_det"
SERVER_REC = "PP-OCRv5_server_rec"


def _gpu_available() -> bool:
    """Check whether a CUDA GPU is available via PaddlePaddle.

    Returns:
        True if at least one CUDA device is available.
    """
    try:
        import paddle

        return paddle.device.cuda.device_count() > 0
    except Exception:
        return False


if _HAS_PADDLEOCR:

    class PaddleOCRModule(OCRAbstract):
        """OCR module using PaddleOCR (PP-OCRv5).

        When a GPU is available the server models and ``device="gpu:0"`` are
        used automatically. On CPU-only machines the lightweight mobile
        models are selected and a warning is logged.
        """

        def __init__(self, config: OCRConfig | dict) -> None:
            """Initialize the PaddleOCR engine.

            Args:
                config: An OCRConfig instance or a dict that will be
                    unpacked into one. Supported model_params keys:
                    ``lang`` (default ``"en"``),
                    ``use_doc_orientation_classify`` (default False),
                    ``use_doc_unwarping`` (default False),
                    ``use_textline_orientation`` (default False).
            """
            super().__init__(config)

            model_params = self.config.model_params

            if _gpu_available():
                device = "gpu:0"
                det_model = model_params.get("text_detection_model_name", SERVER_DET)
                rec_model = model_params.get("text_recognition_model_name", SERVER_REC)
            else:
                device = "cpu"
                det_model = model_params.get("text_detection_model_name", MOBILE_DET)
                rec_model = model_params.get("text_recognition_model_name", MOBILE_REC)
                logger.warning(
                    "No GPU detected -- PaddleOCR will run on CPU with "
                    "lightweight mobile models. Inference will be slower."
                )

            self._reader = PaddleOCR(
                lang=model_params.get("lang", "en"),
                text_detection_model_name=det_model,
                text_recognition_model_name=rec_model,
                device=device,
                use_doc_orientation_classify=model_params.get("use_doc_orientation_classify", False),
                use_doc_unwarping=model_params.get("use_doc_unwarping", False),
                use_textline_orientation=model_params.get("use_textline_orientation", False),
            )

        def _run_single(self, image: np.ndarray, model_params: dict) -> OCRResult:
            """Run PaddleOCR on a single image.

            PaddleOCR does not support per-run parameter overrides. If
            ``model_params`` contains keys that differ from the stored
            config a ValueError is raised.

            Args:
                image: Input image as a numpy array (H x W x C).
                model_params: Must be empty or match the stored config's
                    model_params. Per-run overrides are not supported.

            Returns:
                An OCRResult containing detected text regions.

            Raises:
                ValueError: If model_params contains overrides that differ
                    from the stored configuration.
            """
            if model_params:
                stored = self.config.model_params
                diff = {k: v for k, v in model_params.items() if stored.get(k) != v}
                if diff:
                    raise ValueError(f"PaddleOCR does not support per-run parameter overrides. Differing keys: {diff}")

            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]

            results = self._reader.predict(image)
            bboxes = []
            for res in results:
                polys = res["rec_polys"]
                texts = res["rec_texts"]
                scores = res["rec_scores"]

                for poly, text, score in zip(polys, texts, scores, strict=True):
                    xs = [int(pt[0]) for pt in poly]
                    ys = [int(pt[1]) for pt in poly]
                    bb = BoundingBox(
                        coordinates=((min(xs), min(ys)), (max(xs), max(ys))),
                        text=text,
                        confidence=float(score),
                    )
                    bboxes.append(bb)

            return OCRResult(detections=bboxes)
