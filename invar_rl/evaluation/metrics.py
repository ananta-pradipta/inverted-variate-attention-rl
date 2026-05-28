"""Evaluation metrics, computed per fold.

All portfolio metrics are pure functions of a per-step strategy-return
series and an exposure series; the ranking metric is the mean daily
information coefficient carried on the precomputed tape. No future
information is used: every series is the realised output of a deterministic
rollout.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

_ANN = 252.0


def mean_ic(daily_ic: np.ndarray) -> float:
    """Mean daily information coefficient (Spearman, rank IC by construction)."""
    return float(np.mean(daily_ic)) if daily_ic.size else 0.0


def annualised_sharpe(returns: np.ndarray, ann: float = _ANN) -> float:
    if returns.size < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(ann))


def annualised_sortino(returns: np.ndarray, ann: float = _ANN) -> float:
    if returns.size < 2:
        return 0.0
    downside = returns[returns < 0.0]
    dd = downside.std() if downside.size else 0.0
    if dd == 0.0:
        return 0.0
    return float(returns.mean() / dd * np.sqrt(ann))


def _equity_curve(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + returns)


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown as a non-negative fraction."""
    if returns.size == 0:
        return 0.0
    eq = _equity_curve(returns)
    peak = np.maximum.accumulate(eq)
    return float(np.max(1.0 - eq / peak))


def drawdown_recovery_days(returns: np.ndarray) -> int:
    """Longest run of steps spent below a prior equity peak."""
    if returns.size == 0:
        return 0
    eq = _equity_curve(returns)
    peak = np.maximum.accumulate(eq)
    underwater = eq < peak
    longest = run = 0
    for u in underwater:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return int(longest)


def terminal_wealth(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 1.0
    return float(np.prod(1.0 + returns))


def calmar(returns: np.ndarray, ann: float = _ANN) -> float:
    mdd = max_drawdown(returns)
    if mdd <= 1e-12 or returns.size == 0:
        return 0.0
    ann_return = returns.mean() * ann
    return float(ann_return / mdd)


def conditional_value_at_risk(
    returns: np.ndarray, level: float
) -> float:
    """Left-tail CVaR at ``level`` as a positive loss magnitude."""
    if returns.size < 2:
        return 0.0
    q = np.quantile(returns, level)
    tail = returns[returns <= q]
    return float(-tail.mean()) if tail.size else 0.0


def turnover(exposure: np.ndarray) -> float:
    """Mean absolute day-to-day change in exposure."""
    if exposure.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(exposure))))


def compute_metrics(
    returns: np.ndarray,
    exposure: np.ndarray,
    daily_ic: np.ndarray,
    cvar_level: float = 0.05,
    horizon: int = 1,
) -> Dict[str, float]:
    """All per-fold metrics for one method on one fold.

    The per-step return series is a ``horizon``-day forward return sampled
    once per trading day, so consecutive entries overlap by ``horizon - 1``
    days. Compounding, drawdown, and Sharpe/Sortino/Calmar are therefore
    computed on the **non-overlapping** subsample ``returns[::horizon]`` and
    annualised with ``252 / horizon`` periods per year. This removes the
    overlap-autocorrelation inflation that made the first Phase 6 Sharpe and
    Calmar magnitudes meaningless. With ``horizon = 1`` this reduces exactly
    to the per-step daily convention.

    Per-decision quantities (mean return, turnover, exposure, IC) are
    reported on the full per-step series, since they are not compounded.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    r_ne = returns[::horizon] if returns.size else returns
    ann = _ANN / float(horizon)
    return {
        "ic": mean_ic(daily_ic),
        "rank_ic": mean_ic(daily_ic),
        "mean_return": float(returns.mean()) if returns.size else 0.0,
        "volatility": float(returns.std()) if returns.size else 0.0,
        "sharpe": annualised_sharpe(r_ne, ann),
        "sortino": annualised_sortino(r_ne, ann),
        "max_drawdown": max_drawdown(r_ne),
        "calmar": calmar(r_ne, ann),
        "recovery_days": drawdown_recovery_days(r_ne),
        "terminal_wealth": terminal_wealth(r_ne),
        "cvar": conditional_value_at_risk(r_ne, cvar_level),
        "turnover": turnover(exposure),
        "mean_exposure": float(exposure.mean()) if exposure.size else 0.0,
        "net_exposure": float(exposure.mean()) if exposure.size else 0.0,
    }
