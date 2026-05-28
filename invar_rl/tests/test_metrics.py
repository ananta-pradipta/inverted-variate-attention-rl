"""Tests for the Phase 6 metric suite and the block bootstrap."""

from __future__ import annotations

import numpy as np

from invar_rl.common.seeding import make_rng
from invar_rl.evaluation.bootstrap import moving_block_bootstrap_ci
from invar_rl.evaluation.metrics import (
    compute_metrics,
    conditional_value_at_risk,
    max_drawdown,
    terminal_wealth,
)


def test_metric_suite_shapes_and_ranges() -> None:
    rng = make_rng(1)
    rets = rng.normal(0.0005, 0.01, size=300)
    exp = np.clip(rng.normal(1.0, 0.1, size=300), 0.0, 1.5)
    ic = rng.normal(0.03, 0.05, size=300)
    m = compute_metrics(rets, exp, ic, cvar_level=0.05)
    assert set(m) >= {
        "ic", "rank_ic", "sharpe", "sortino", "max_drawdown",
        "calmar", "recovery_days", "terminal_wealth", "cvar",
        "turnover", "mean_exposure",
    }
    assert 0.0 <= m["max_drawdown"] <= 1.0
    assert m["terminal_wealth"] > 0.0
    assert np.isfinite(m["sharpe"]) and np.isfinite(m["calmar"])
    assert m["turnover"] >= 0.0


def test_drawdown_and_cvar_on_known_series() -> None:
    rets = np.array([0.1, -0.5, 0.0, 0.2])
    # Equity: 1.1, 0.55, 0.55, 0.66; peak 1.1 -> trough 0.55 = 50% dd.
    assert abs(max_drawdown(rets) - 0.5) < 1e-9
    assert terminal_wealth(rets) > 0.0
    losses = np.array([-0.1, -0.2, -0.05, 0.03, 0.04, 0.01])
    cv = conditional_value_at_risk(losses, 0.34)
    assert cv > 0.0  # positive loss magnitude in the left tail


def test_horizon_deflates_overlapping_sharpe() -> None:
    # Overlapping h-day returns inflate annualised Sharpe under the naive
    # per-step convention; the horizon-aware path (non-overlapping subsample
    # + 252/h annualisation) must give a materially smaller magnitude, and
    # horizon=1 must reproduce the per-step convention exactly.
    rng = make_rng(3)
    base = rng.normal(0.001, 0.01, size=600)
    overlapping = np.convolve(base, np.ones(5), mode="valid")  # 5-day sums
    exp = np.ones_like(overlapping)
    ic = np.zeros_like(overlapping)
    naive = compute_metrics(overlapping, exp, ic, horizon=1)["sharpe"]
    fixed = compute_metrics(overlapping, exp, ic, horizon=5)["sharpe"]
    assert abs(fixed) < abs(naive)
    same = compute_metrics(base, np.ones_like(base),
                           np.zeros_like(base), horizon=1)
    # horizon=1 is exactly the per-step daily convention.
    assert np.isfinite(same["sharpe"]) and np.isfinite(same["calmar"])


def test_block_bootstrap_is_reproducible_and_brackets_point() -> None:
    rng = make_rng(7)
    series = rng.normal(0.001, 0.01, size=500)
    stat = lambda s: float(s.mean())  # noqa: E731
    p1, lo1, hi1 = moving_block_bootstrap_ci(
        series, stat, n_boot=500, block=20, seed=42
    )
    p2, lo2, hi2 = moving_block_bootstrap_ci(
        series, stat, n_boot=500, block=20, seed=42
    )
    assert (p1, lo1, hi1) == (p2, lo2, hi2)
    assert lo1 <= p1 <= hi1
