import cv2
import numpy as np

from ocr_backbone.input_image import InputImage

BINARIZE_METHODS = ("adaptive", "otsu")


def crop_image(input_image: InputImage, boundaries: tuple[tuple[int, int], tuple[int, int]]) -> InputImage:
    """Crop an InputImage to the given boundaries, accumulating offsets.

    Args:
        input_image: The source image to crop.
        boundaries: A ((x_min, y_min), (x_max, y_max)) tuple defining the
            crop region relative to the input_image array.

    Returns:
        A new InputImage containing the cropped sub-image with updated offsets.

    Raises:
        ValueError: If boundaries extend outside the image dimensions.
    """
    (x_min, y_min), (x_max, y_max) = boundaries
    h, w = input_image.image.shape[:2]
    if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
        raise ValueError(
            f"Boundaries {boundaries} exceed image dimensions ({w}, {h})."
        )
    cropped = input_image.image[y_min:y_max, x_min:x_max]
    return InputImage(
        image=cropped,
        x_offset=input_image.x_offset + x_min,
        y_offset=input_image.y_offset + y_min,
    )


def binarize(input_image: InputImage, method: str = "adaptive", block_size: int = 11, constant: int = 2) -> InputImage:
    """Convert an InputImage to a binary (black and white) image.

    Converts to grayscale first if the input is a color image, then applies
    thresholding.

    Args:
        input_image: The source image to binarize.
        method: Thresholding method. "adaptive" for Gaussian adaptive
            thresholding, "otsu" for global Otsu's threshold.
        block_size: Size of the pixel neighbourhood used for adaptive
            thresholding. Must be an odd integer greater than 1.
            Ignored when method is "otsu".
        constant: Constant subtracted from the adaptive threshold mean.
            Ignored when method is "otsu".

    Returns:
        A new InputImage containing the binarized image (single channel).

    Raises:
        ValueError: If method is not one of the supported methods.
    """
    if method not in BINARIZE_METHODS:
        raise ValueError(
            f"Unknown binarization method '{method}'. Supported: {BINARIZE_METHODS}."
        )

    image = input_image.image
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.squeeze() if len(image.shape) == 3 else image

    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant,
        )
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return InputImage(
        image=binary,
        x_offset=input_image.x_offset,
        y_offset=input_image.y_offset,
    )


def grid_split_image(input_image: InputImage, grid: tuple[int,int])-> list[InputImage]:
        """Split an InputImage into a grid of sub-images.

        Args:
            input_image: The source image to split.
            grid: A (rows, cols) tuple defining the grid dimensions.

        Returns:
            A list (rows x cols) of sub-image InputImages.
        """
        rows = grid[0]
        cols = grid[1]
        h, w = input_image.image.shape[:2]
        row_edges = np.linspace(0, h, rows + 1, dtype=int)
        col_edges = np.linspace(0, w, cols + 1, dtype=int)
        cells = []
        for r in range(rows):
            for c in range(cols):
                boundaries = (
                    (int(col_edges[c]), int(row_edges[r])),
                    (int(col_edges[c + 1]), int(row_edges[r + 1])),
                )
                cells.append(crop_image(input_image, boundaries))

        return cells


def contour_split_image(input_image: InputImage) -> list[InputImage]:
    """Split an InputImage into sub-images by detecting natural contour boundaries.

    Converts the image to greyscale, applies Sobel edge detection to find
    intensity boundaries, then extracts each detected contour region as a
    separate sub-image. Pixels outside the contour but inside its bounding
    rectangle are zero-padded.

    Args:
        input_image: The source image to split.

    Returns:
        A list of InputImage fragments, one per detected external contour.
        Each fragment's image is the bounding rectangle of its contour with
        pixels outside the contour zeroed out. Offsets reflect the fragment's
        position in the original image coordinate space.
    """
    image = input_image.image
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.squeeze() if len(image.shape) == 3 else image

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8) if magnitude.max() > 0 else magnitude.astype(np.uint8)

    _, binary = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fragments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        mask_crop = mask[y:y + h, x:x + w]

        if len(image.shape) == 3:
            region = image[y:y + h, x:x + w].copy()
            region[mask_crop == 0] = 0
        else:
            region = gray[y:y + h, x:x + w].copy()
            region[mask_crop == 0] = 0

        fragments.append(InputImage(
            image=region,
            x_offset=input_image.x_offset + x,
            y_offset=input_image.y_offset + y,
        ))

    return fragments
