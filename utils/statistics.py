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
    """Compute a confidence interval for a metric bounded to [0, 1].

    A Beta distribution is fit to the observations to derive the
    interval. Values at the exact boundaries (0 or 1) are nudged
    inward by ``eps`` so the fit succeeds. When every observation
    collapses to a single value the Beta fit is not identifiable;
    in that degenerate case the function falls back to a Wilson
    score interval treating the observed mean as a Bernoulli
    proportion over ``len(values)`` trials, with the result clipped
    to [0, 1].

    Args:
        values: Array of metric values in [0, 1].
        level: Confidence level as a percentage (e.g. 95).

    Returns:
        A two-element list [lower_bound, upper_bound].
    """
    # Smithson-Verkuilen boundary nudge: clamp distance from {0, 1} scales as 0.5/n
    # so boundary observations stay away from the digamma singularity. A fixed
    # tiny eps (e.g. 1e-6) makes log(x) and log(1-x) explode for small samples
    # with mass at the edges, which drives scipy's Beta MLE into a region where
    # the solver fails to converge (FitSolverError).
    eps = 0.5 / len(values)
    clamped = np.clip(values, eps, 1 - eps)
    if not np.all(clamped == clamped[0]):
        a, b, _, _ = _stats.beta.fit(clamped, floc=0, fscale=1)
        lo = (100 - level) / 200
        hi = (100 + level) / 200
        return [float(_stats.beta.ppf(lo, a, b)), float(_stats.beta.ppf(hi, a, b))]

    # Wilson score confidence interval
    alpha = 1 - level / 100
    z = float(_stats.norm.ppf(1 - alpha / 2))
    denom = 1 + z**2 / len(values)
    p_hat = float(np.mean(values))
    center = (p_hat + z**2 / (2 * len(values))) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / len(values) + z**2 / (4 * len(values) ** 2)) / denom
    return [float(np.clip(center - half, 0.0, 1.0)), float(np.clip(center + half, 0.0, 1.0))]


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
