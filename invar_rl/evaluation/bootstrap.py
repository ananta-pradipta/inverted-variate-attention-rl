"""Block-bootstrap confidence intervals for autocorrelated daily series.

A moving-block bootstrap preserves short-range autocorrelation in the daily
return series, which the i.i.d. bootstrap would destroy. Used to put
confidence intervals on every fold-level metric, with emphasis on the
out-of-distribution stress fold.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def moving_block_bootstrap_ci(
    series: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    block: int = 20,
    level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Confidence interval for ``stat_fn`` via a moving-block bootstrap.

    Args:
        series: The 1-D daily series (for example strategy returns).
        stat_fn: Maps a resampled series to a scalar statistic.
        n_boot: Number of bootstrap resamples.
        block: Block length in days; should exceed the autocorrelation
            horizon.
        level: Two-sided confidence level.
        seed: RNG seed for reproducibility.

    Returns:
        A triple ``(point, lo, hi)``: the statistic on the original series
        and the lower and upper percentile bounds.
    """
    series = np.asarray(series, dtype=np.float64)
    n = series.size
    point = float(stat_fn(series))
    if n < 2:
        return point, point, point

    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    max_start = n - block

    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate(
            [series[s : s + block] for s in starts]
        )[:n]
        stats[b] = stat_fn(sample)

    alpha = (1.0 - level) / 2.0
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1.0 - alpha))
    return point, lo, hi
