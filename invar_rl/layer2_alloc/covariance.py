"""Covariance estimators for the allocation layer.

Two estimators of the stock-return covariance matrix:

1. Ledoit-Wolf shrinkage toward a structured constant-correlation target,
   with the analytically optimal shrinkage intensity.
2. A low-rank factor-model covariance plus a diagonal idiosyncratic term.

Both are pure functions of the returns matrix passed in. The caller is
responsible for passing training-fold returns only, with a real-time
convention, so no future, validation, or test information enters the
estimate. The estimator is selected by configuration.
"""

from __future__ import annotations

import numpy as np

_JITTER = 1e-8


def _as_2d(returns: np.ndarray) -> np.ndarray:
    if returns.ndim != 2:
        raise ValueError(
            f"returns must be (T, N), got shape {returns.shape}"
        )
    if returns.shape[0] < 2:
        raise ValueError("need at least two observations to estimate covariance")
    return np.asarray(returns, dtype=np.float64)


def ledoit_wolf_constant_correlation(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward a constant-correlation target.

    Implements the Ledoit and Wolf (2004) honey-I-shrunk estimator: the
    sample covariance is shrunk toward a structured target that keeps the
    sample variances but replaces every pairwise correlation with the
    average sample correlation. The shrinkage intensity is the analytic
    optimum, clipped to [0, 1].

    Args:
        returns: Training-fold returns, shape (T, N).

    Returns:
        A symmetric positive-definite covariance matrix, shape (N, N).
    """
    x = _as_2d(returns)
    t, n = x.shape
    x = x - x.mean(axis=0, keepdims=True)
    sample = (x.T @ x) / t

    var = np.diag(sample)
    std = np.sqrt(np.maximum(var, _JITTER))
    outer_std = np.outer(std, std)
    corr = sample / outer_std
    off = ~np.eye(n, dtype=bool)
    r_bar = corr[off].mean() if n > 1 else 0.0

    target = r_bar * outer_std
    np.fill_diagonal(target, var)

    # Optimal shrinkage intensity (Ledoit-Wolf 2004).
    y = x ** 2
    pi_mat = (y.T @ y) / t - sample ** 2
    pi_hat = pi_mat.sum()

    rho_diag = np.diag(pi_mat).sum()
    term = (x ** 3).T @ x / t - var[:, None] * sample
    np.fill_diagonal(term, 0.0)
    rho_off = (
        r_bar
        * (1.0 / std[:, None] * std[None, :] * term).sum()
    )
    rho_hat = rho_diag + rho_off

    gamma_hat = np.linalg.norm(sample - target, "fro") ** 2
    if gamma_hat <= 0.0:
        kappa = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
    delta = float(np.clip(kappa / t, 0.0, 1.0))

    shrunk = delta * target + (1.0 - delta) * sample
    shrunk = 0.5 * (shrunk + shrunk.T)
    shrunk[np.diag_indices(n)] += _JITTER
    return shrunk


def factor_model_covariance(
    returns: np.ndarray, n_factors: int
) -> np.ndarray:
    """Low-rank factor covariance plus a diagonal idiosyncratic term.

    Estimates the leading principal components of the centred returns,
    reconstructs the systematic covariance from the top ``n_factors``
    components, and adds the diagonal of the residual covariance so the
    result is positive-definite.

    Args:
        returns: Training-fold returns, shape (T, N).
        n_factors: Number of retained factors (1 <= n_factors < N).

    Returns:
        A symmetric positive-definite covariance matrix, shape (N, N).
    """
    x = _as_2d(returns)
    t, n = x.shape
    if not 1 <= n_factors < n:
        raise ValueError(
            f"n_factors must be in [1, N-1]={n - 1}, got {n_factors}"
        )
    x = x - x.mean(axis=0, keepdims=True)
    sample = (x.T @ x) / t

    eigvals, eigvecs = np.linalg.eigh(sample)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    load = eigvecs[:, :n_factors] * np.sqrt(eigvals[:n_factors])
    systematic = load @ load.T
    idio = np.maximum(np.diag(sample) - np.diag(systematic), _JITTER)

    cov = systematic + np.diag(idio)
    cov = 0.5 * (cov + cov.T)
    cov[np.diag_indices(n)] += _JITTER
    return cov


def estimate_covariance(
    returns: np.ndarray, estimator: str, factor_rank: int
) -> np.ndarray:
    """Dispatch to the configured covariance estimator.

    Args:
        returns: Training-fold returns, shape (T, N).
        estimator: Either "ledoit_wolf" or "factor_model".
        factor_rank: Factor count used when ``estimator`` is "factor_model".

    Returns:
        The estimated covariance matrix, shape (N, N).

    Raises:
        ValueError: If ``estimator`` is unknown.
    """
    if estimator == "ledoit_wolf":
        return ledoit_wolf_constant_correlation(returns)
    if estimator == "factor_model":
        return factor_model_covariance(returns, factor_rank)
    raise ValueError(
        f"unknown estimator {estimator!r}, expected 'ledoit_wolf' or "
        "'factor_model'"
    )
