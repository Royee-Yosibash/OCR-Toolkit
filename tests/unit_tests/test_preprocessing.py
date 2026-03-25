import unittest

import numpy as np

from ocr_backbone.image_preprocessing import binarize, crop_image, grid_split_image
from ocr_backbone.input_image import InputImage


class TestGridSplitImage(unittest.TestCase):
    """Tests for grid_split_image and related preprocessing utilities."""

    def test_single_cell_returns_whole_image(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        cells = grid_split_image(InputImage(image=image), grid=(1, 1))
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].image.shape, (100, 200, 3))
        self.assertEqual(cells[0].x_offset, 0)
        self.assertEqual(cells[0].y_offset, 0)

    def test_2x2_grid_dimensions_and_offsets(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        cells = grid_split_image(InputImage(image=image), grid=(2, 2))
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[0].image.shape[:2], (50, 100))
        self.assertEqual((cells[0].x_offset, cells[0].y_offset), (0, 0))
        self.assertEqual((cells[1].x_offset, cells[1].y_offset), (100, 0))
        self.assertEqual((cells[2].x_offset, cells[2].y_offset), (0, 50))
        self.assertEqual((cells[3].x_offset, cells[3].y_offset), (100, 50))

    def test_3x3_grid_cell_dimensions(self):
        image = np.zeros((90, 120, 3), dtype=np.uint8)
        cells = grid_split_image(InputImage(image=image), grid=(3, 3))
        self.assertEqual(len(cells), 9)
        for cell in cells:
            self.assertEqual(cell.image.shape[0], 30)
            self.assertEqual(cell.image.shape[1], 40)

    def test_single_row_grid(self):
        image = np.zeros((50, 120, 3), dtype=np.uint8)
        cells = grid_split_image(InputImage(image=image), grid=(1, 3))
        self.assertEqual(len(cells), 3)
        for cell in cells:
            self.assertEqual(cell.image.shape[:2], (50, 40))
        self.assertEqual(cells[0].x_offset, 0)
        self.assertEqual(cells[1].x_offset, 40)
        self.assertEqual(cells[2].x_offset, 80)

    def test_single_column_grid(self):
        image = np.zeros((90, 50, 3), dtype=np.uint8)
        cells = grid_split_image(InputImage(image=image), grid=(3, 1))
        self.assertEqual(len(cells), 3)
        for cell in cells:
            self.assertEqual(cell.image.shape[:2], (30, 50))
        self.assertEqual(cells[0].y_offset, 0)
        self.assertEqual(cells[1].y_offset, 30)
        self.assertEqual(cells[2].y_offset, 60)

    def test_non_evenly_divisible_dimensions(self):
        """Linspace-based splitting should handle remainder pixels."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cells = grid_split_image(InputImage(image=image), grid=(3, 3))
        self.assertEqual(len(cells), 9)
        total_h = sum(cells[i].image.shape[0] for i in range(0, 9, 3))
        total_w = sum(cells[i].image.shape[1] for i in range(3))
        self.assertEqual(total_h, 100)
        self.assertEqual(total_w, 100)

    def test_nonzero_initial_offset_accumulates(self):
        """Offsets from the InputImage should be added to cell offsets."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        input_image = InputImage(image=image, x_offset=50, y_offset=30)
        cells = grid_split_image(input_image, grid=(2, 2))
        self.assertEqual((cells[0].x_offset, cells[0].y_offset), (50, 30))
        self.assertEqual((cells[1].x_offset, cells[1].y_offset), (150, 30))
        self.assertEqual((cells[2].x_offset, cells[2].y_offset), (50, 80))
        self.assertEqual((cells[3].x_offset, cells[3].y_offset), (150, 80))


class TestCropImage(unittest.TestCase):
    """Tests for crop_image functionality."""

    def test_crop_dimensions(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = crop_image(InputImage(image=image), boundaries=((50, 20), (150, 80)))
        self.assertEqual(result.image.shape, (60, 100, 3))

    def test_offsets_from_zero(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = crop_image(InputImage(image=image), boundaries=((50, 20), (150, 80)))
        self.assertEqual(result.x_offset, 50)
        self.assertEqual(result.y_offset, 20)

    def test_offsets_accumulate(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        input_image = InputImage(image=image, x_offset=10, y_offset=5)
        result = crop_image(input_image, boundaries=((50, 20), (150, 80)))
        self.assertEqual(result.x_offset, 60)
        self.assertEqual(result.y_offset, 25)

    def test_full_image_boundaries(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = crop_image(InputImage(image=image), boundaries=((0, 0), (200, 100)))
        self.assertEqual(result.image.shape, (100, 200, 3))
        self.assertEqual(result.x_offset, 0)
        self.assertEqual(result.y_offset, 0)

    def test_boundaries_exceed_image_raises(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            crop_image(InputImage(image=image), boundaries=((0, 0), (300, 100)))

    def test_pixel_content_preserved(self):
        image = np.arange(60, dtype=np.uint8).reshape((6, 10, 1))
        result = crop_image(InputImage(image=image), boundaries=((2, 1), (5, 4)))
        expected = image[1:4, 2:5]
        np.testing.assert_array_equal(result.image, expected)


class TestBinarize(unittest.TestCase):
    """Tests for binarize functionality."""

    def test_output_is_binary(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = binarize(InputImage(image=image))
        unique = set(np.unique(result.image))
        self.assertTrue(unique.issubset({0, 255}))

    def test_output_is_single_channel(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        result = binarize(InputImage(image=image))
        self.assertEqual(len(result.image.shape), 2)

    def test_grayscale_input(self):
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        result = binarize(InputImage(image=gray))
        self.assertEqual(len(result.image.shape), 2)
        unique = set(np.unique(result.image))
        self.assertTrue(unique.issubset({0, 255}))

    def test_preserves_offsets(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        result = binarize(InputImage(image=image, x_offset=10, y_offset=20))
        self.assertEqual(result.x_offset, 10)
        self.assertEqual(result.y_offset, 20)

    def test_preserves_dimensions(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        result = binarize(InputImage(image=image))
        self.assertEqual(result.image.shape, (80, 120))

    def test_otsu_method(self):
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = binarize(InputImage(image=image), method="otsu")
        unique = set(np.unique(result.image))
        self.assertTrue(unique.issubset({0, 255}))

    def test_unknown_method_raises(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            binarize(InputImage(image=image), method="unknown")

    def test_white_image_stays_white(self):
        image = np.full((50, 50, 3), 255, dtype=np.uint8)
        result = binarize(InputImage(image=image))
        np.testing.assert_array_equal(result.image, np.full((50, 50), 255, dtype=np.uint8))
