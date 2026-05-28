"""Stationary block bootstrap for the Sharpe ratio.

Implements the Politis-Romano (1994) stationary bootstrap with a geometric
block-length distribution, applied to daily portfolio returns. The Sharpe
ratio standard error and a percentile 95% CI are returned.

Annualization assumes 252 trading days per year, matching the canonical
InVAR-RL evaluation convention. No silent fallbacks: an empty or singleton
input raises ValueError; a NaN input raises ValueError.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

ANNUAL_FACTOR: float = float(np.sqrt(252.0))


def _sharpe_annualized(returns: np.ndarray) -> float:
    """Compute the annualized Sharpe ratio of a daily-return series.

    Args:
        returns: One-dimensional array of per-day returns.

    Returns:
        Annualized Sharpe (mean / std(ddof=1) * sqrt(252)). Returns 0.0
        when the standard deviation is below 1e-12 to avoid divide-by-zero
        while still propagating the signal in normal regimes.
    """
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    if std < 1e-12:
        return 0.0
    return mean / std * ANNUAL_FACTOR


def _stationary_block_sample(
    returns: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a single stationary-bootstrap resample.

    Block lengths are geometrically distributed with mean `block_length`;
    start indices are uniform over `[0, n)`; the source series is wrapped
    circularly so every position is sampled with equal marginal probability.

    Args:
        returns: One-dimensional array of per-day returns (length n).
        block_length: Mean geometric block length L; the per-step
            continuation probability is `1 - 1 / L`.
        rng: Numpy random Generator instance.

    Returns:
        Resampled return series of the same length as `returns`.
    """
    n = returns.shape[0]
    if block_length < 1:
        raise ValueError(
            f"[ERR] block_length must be >= 1; got {block_length}"
        )
    p_continue = 1.0 - 1.0 / float(block_length)
    out = np.empty(n, dtype=returns.dtype)
    idx = int(rng.integers(0, n))
    for t in range(n):
        out[t] = returns[idx]
        if rng.random() < p_continue:
            idx = (idx + 1) % n
        else:
            idx = int(rng.integers(0, n))
    return out


def stationary_block_bootstrap_sharpe_se(
    daily_returns: np.ndarray,
    block_length: int = 5,
    n_replications: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute Sharpe-ratio standard error via stationary block bootstrap.

    Args:
        daily_returns: One-dimensional array of per-day portfolio returns.
        block_length: Mean geometric block length (Politis-Romano L).
        n_replications: Number of bootstrap resamples.
        seed: Seed for the bootstrap RNG (does not affect input data).

    Returns:
        Dict with keys: `sharpe_point`, `sharpe_se`, `sharpe_ci_95_low`,
        `sharpe_ci_95_high`.

    Raises:
        ValueError: if input is empty, has fewer than 2 entries, contains
            NaN, or if block_length / n_replications are non-positive.
    """
    arr = np.asarray(daily_returns, dtype=np.float64).ravel()
    if arr.size < 2:
        raise ValueError(
            f"[ERR] daily_returns must have >= 2 entries; got {arr.size}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(
            "[ERR] daily_returns contains NaN or inf"
        )
    if n_replications < 1:
        raise ValueError(
            f"[ERR] n_replications must be >= 1; got {n_replications}"
        )

    rng = np.random.default_rng(seed)
    point = _sharpe_annualized(arr)

    samples = np.empty(n_replications, dtype=np.float64)
    for r in range(n_replications):
        resample = _stationary_block_sample(arr, block_length, rng)
        samples[r] = _sharpe_annualized(resample)

    se = float(samples.std(ddof=1))
    ci_low = float(np.percentile(samples, 2.5))
    ci_high = float(np.percentile(samples, 97.5))

    return {
        "sharpe_point": float(point),
        "sharpe_se": se,
        "sharpe_ci_95_low": ci_low,
        "sharpe_ci_95_high": ci_high,
    }
