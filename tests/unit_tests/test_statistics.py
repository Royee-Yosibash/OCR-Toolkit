import unittest

import numpy as np

from utils.statistics import (
    beta_ci,
    bootstrap_ci,
    BOOTSTRAP_MIN_SAMPLE_SIZE,
)


class TestBetaCI(unittest.TestCase):
    """Tests for beta_ci confidence interval estimation."""

    def test_symmetric_distribution(self):
        """Verify CI straddles 0.5 for a symmetric Beta(5,5) distribution."""
        rng = np.random.default_rng(0)
        values = rng.beta(5, 5, size=500)
        lo, hi = beta_ci(values, 95)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertGreater(lo, 0.0)
        self.assertLess(hi, 1.0)

    def test_narrow_ci_for_tight_distribution(self):
        """Verify CI is narrow when variance is very small."""
        rng = np.random.default_rng(7)
        values = rng.beta(500, 500, size=200)
        lo, hi = beta_ci(values, 95)
        self.assertAlmostEqual(lo, 0.5, places=1)
        self.assertAlmostEqual(hi, 0.5, places=1)

    def test_boundary_values_clamped(self):
        """Verify exact 0 and 1 values are handled without crashing."""
        values = np.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.5])
        lo, hi = beta_ci(values, 95)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_higher_level_wider_interval(self):
        """Verify a 99% CI is at least as wide as a 90% CI."""
        rng = np.random.default_rng(1)
        values = rng.beta(2, 5, size=300)
        lo_90, hi_90 = beta_ci(values, 90)
        lo_99, hi_99 = beta_ci(values, 99)
        self.assertGreaterEqual(lo_90, lo_99)
        self.assertLessEqual(hi_90, hi_99)

    def test_returns_two_element_list(self):
        """Verify the return type is a two-element list of floats."""
        values = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        result = beta_ci(values, 95)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_lower_bound_less_than_upper(self):
        """Verify the lower bound is strictly less than the upper bound."""
        rng = np.random.default_rng(2)
        values = rng.beta(2, 3, size=100)
        lo, hi = beta_ci(values, 95)
        self.assertLess(lo, hi)


class TestBootstrapCI(unittest.TestCase):
    """Tests for bootstrap_ci confidence interval estimation."""

    def test_percentile_fallback_small_sample(self):
        """Verify percentile fallback is used when sample is below the minimum size."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertLess(len(values), BOOTSTRAP_MIN_SAMPLE_SIZE)
        lo, hi = bootstrap_ci(values, 90)
        self.assertGreaterEqual(lo, 1.0)
        self.assertLessEqual(hi, 5.0)

    def test_bootstrap_with_large_sample(self):
        """Verify bootstrap resampling path runs when sample exceeds minimum size."""
        rng = np.random.default_rng(3)
        values = rng.normal(10.0, 2.0, size=200)
        self.assertGreaterEqual(len(values), BOOTSTRAP_MIN_SAMPLE_SIZE)
        lo, hi = bootstrap_ci(values, 95)
        self.assertLess(lo, 10.0)
        self.assertGreater(hi, 10.0)

    def test_returns_two_element_list(self):
        """Verify the return type is a two-element list of floats."""
        values = np.array([1.0, 2.0, 3.0])
        result = bootstrap_ci(values, 95)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_lower_bound_less_than_upper(self):
        """Verify the lower bound is strictly less than the upper bound."""
        rng = np.random.default_rng(4)
        values = rng.normal(5.0, 1.0, size=150)
        lo, hi = bootstrap_ci(values, 95)
        self.assertLess(lo, hi)

    def test_higher_level_wider_interval(self):
        """Verify a 99% CI is at least as wide as a 90% CI."""
        rng = np.random.default_rng(5)
        values = rng.normal(0.0, 1.0, size=200)
        lo_90, hi_90 = bootstrap_ci(values, 90)
        lo_99, hi_99 = bootstrap_ci(values, 99)
        self.assertGreaterEqual(lo_90, lo_99)
        self.assertLessEqual(hi_90, hi_99)

    def test_constant_values_narrow_interval(self):
        """Verify CI collapses to the constant value when all inputs are identical."""
        values = np.full(150, 7.0)
        lo, hi = bootstrap_ci(values, 95)
        self.assertAlmostEqual(lo, 7.0, places=5)
        self.assertAlmostEqual(hi, 7.0, places=5)

    def test_deterministic_with_seed(self):
        """Verify repeated calls produce identical results due to fixed RNG seed."""
        rng = np.random.default_rng(6)
        values = rng.normal(0.0, 1.0, size=200)
        result_a = bootstrap_ci(values, 95)
        result_b = bootstrap_ci(values, 95)
        self.assertEqual(result_a, result_b)
