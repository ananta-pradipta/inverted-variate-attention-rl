"""Tests for Phase 3 OnlineSharpeReward."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.robust_invar_rl.online_sharpe_reward import (
    OnlineSharpeReward,
    OnlineSharpeRewardConfig,
    TRADING_DAYS_PER_YEAR,
)


def test_alpha_matches_half_life_formula() -> None:
    cfg = OnlineSharpeRewardConfig(half_life_days=21)
    reward = OnlineSharpeReward(cfg=cfg)
    expected = 1.0 - 0.5 ** (1.0 / 21.0)
    assert reward.alpha == pytest.approx(expected, rel=1e-9)


def test_warmup_returns_scaled_raw_return() -> None:
    cfg = OnlineSharpeRewardConfig(warmup_steps=3)
    reward = OnlineSharpeReward(cfg=cfg)
    reward.reset()
    r1 = reward.step(0.01)
    assert r1 == pytest.approx(
        0.01 * float(np.sqrt(TRADING_DAYS_PER_YEAR)), rel=1e-6
    )


def test_post_warmup_returns_finite_scalar() -> None:
    cfg = OnlineSharpeRewardConfig(warmup_steps=2)
    reward = OnlineSharpeReward(cfg=cfg)
    reward.reset()
    # 2 warmup steps + 1 real step.
    reward.step(0.01)
    reward.step(-0.005)
    r3 = reward.step(0.003)
    assert isinstance(r3, float)
    assert np.isfinite(r3)


def test_reward_is_clipped() -> None:
    cfg = OnlineSharpeRewardConfig(warmup_steps=0, clip=2.0)
    reward = OnlineSharpeReward(cfg=cfg)
    reward.reset()
    # A massive return on first step (warmup=0) goes through the EWMA
    # path with effectively zero prior var -> sigma_eff = eps -> giant.
    r = reward.step(0.5)
    assert abs(r) <= 2.0 + 1e-6


def test_reset_clears_state() -> None:
    cfg = OnlineSharpeRewardConfig(warmup_steps=1)
    reward = OnlineSharpeReward(cfg=cfg)
    reward.reset()
    reward.step(0.01)
    reward.step(0.02)
    assert reward.n_steps == 2
    reward.reset()
    assert reward.n_steps == 0
    assert reward.mu == 0.0
    assert reward.var == 0.0


def test_step_raises_on_nan_input() -> None:
    reward = OnlineSharpeReward()
    reward.reset()
    with pytest.raises(ValueError, match="non-finite"):
        reward.step(np.nan)


def test_config_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        OnlineSharpeRewardConfig(half_life_days=0)
    with pytest.raises(ValueError):
        OnlineSharpeRewardConfig(eps=0.0)
    with pytest.raises(ValueError):
        OnlineSharpeRewardConfig(warmup_steps=-1)
    with pytest.raises(ValueError):
        OnlineSharpeRewardConfig(clip=0.0)


def test_constant_return_stream_produces_zero_mean_after_warmup() -> None:
    """If r is constant, mu_new == mu_old, so Sharpe increment is 0."""
    cfg = OnlineSharpeRewardConfig(warmup_steps=5)
    reward = OnlineSharpeReward(cfg=cfg)
    reward.reset()
    # 5 warm-up + many steady-state steps with identical r.
    for _ in range(50):
        reward.step(0.001)
    # After warmup, additional constant returns should yield ~0 reward.
    final = reward.step(0.001)
    assert abs(final) < 1.0  # close to zero (post-warmup, near-steady)
