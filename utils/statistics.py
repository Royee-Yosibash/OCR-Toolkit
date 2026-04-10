"""Statistical utilities for confidence interval estimation.

Provides Beta-distribution and bootstrap-based confidence interval
computation for bounded and unbounded metric distributions respectively.
"""

import logging

import numpy as np

from utils.lazy_import import LazyModule

logger = logging.getLogger(__name__)

_stats = LazyModule("scipy.stats")

BOOTSTRAP_N_RESAMPLES = 10000
BOOTSTRAP_MIN_SAMPLE_SIZE = 100


def beta_ci(values: np.ndarray, level: int) -> list[float]:
    """Compute a confidence interval using a Beta distribution fit.

    Suitable for metrics bounded to [0, 1]. Values at the exact
    boundaries (0 or 1) are nudged inward so the Beta fit succeeds.

    Args:
        values: Array of metric values in [0, 1].
        level: Confidence level as a percentage (e.g. 95).

    Returns:
        A two-element list [lower_bound, upper_bound].
    """
    eps = 1e-6
    clamped = np.clip(values, eps, 1 - eps)
    a, b, _, _ = _stats.beta.fit(clamped, floc=0, fscale=1)
    lo = (100 - level) / 200
    hi = (100 + level) / 200
    return [float(_stats.beta.ppf(lo, a, b)), float(_stats.beta.ppf(hi, a, b))]


def bootstrap_ci(values: np.ndarray, level: int) -> list[float]:
    """Compute a confidence interval for unbounded metrics.

    Uses bootstrap resampling when the sample size exceeds
    ``BOOTSTRAP_MIN_SAMPLE_SIZE``, otherwise falls back to direct
    percentiles of the observed values.

    Args:
        values: Array of metric values.
        level: Confidence level as a percentage (e.g. 95).

    Returns:
        A two-element list [lower_bound, upper_bound].
    """
    lo = (100 - level) / 2
    hi = (100 + level) / 2

    if len(values) < BOOTSTRAP_MIN_SAMPLE_SIZE:
        logger.info(
            "Sample size %d is below %d -- using percentile CI instead of bootstrap.",
            len(values),
            BOOTSTRAP_MIN_SAMPLE_SIZE,
        )
        means = values
    else:
        rng = np.random.default_rng(42)
        means = np.array(
            [rng.choice(values, size=len(values), replace=True).mean() for _ in range(BOOTSTRAP_N_RESAMPLES)]
        )

    return [float(np.percentile(means, lo)), float(np.percentile(means, hi))]
