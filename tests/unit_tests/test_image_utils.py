import unittest

import numpy as np

from utils.image_utils import to_rgb


class TestToRGB(unittest.TestCase):
    """Tests for :func:`utils.image_utils.to_rgb`."""

    def test_grayscale_2d_input_is_broadcast(self):
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        result = to_rgb(image)
        self.assertEqual(result.shape, (3, 4, 3))
        for c in range(3):
            np.testing.assert_array_equal(result[:, :, c], image)

    def test_rgb_input_is_unchanged(self):
        image = np.zeros((5, 6, 3), dtype=np.uint8)
        image[0, 0] = [10, 20, 30]
        result = to_rgb(image)
        self.assertIs(result, image)

    def test_rgba_input_drops_alpha(self):
        image = np.zeros((4, 5, 4), dtype=np.uint8)
        image[..., 3] = 99
        image[0, 0, :3] = [1, 2, 3]
        result = to_rgb(image)
        self.assertEqual(result.shape, (4, 5, 3))
        np.testing.assert_array_equal(result[0, 0], [1, 2, 3])

    def test_unsupported_channel_count_raises(self):
        image = np.zeros((3, 4, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            to_rgb(image)

    def test_unsupported_ndim_raises(self):
        image = np.zeros((2, 3, 4, 1), dtype=np.uint8)
        with self.assertRaises(ValueError):
            to_rgb(image)
