"""Tests for Phase 3 CompactObservationWrapper + OnlineSharpeRewardWrapper.

We construct a stub of :class:`KellyPriorEnvWrapper` so the test does
not require the full lattice bridge / InVAR ckpt. The stub publishes
the same ``info`` keys (``e_star``, ``residual_action``, ``e_final``,
``strategy_return``) and the same 1-D ``Box([-1, +1])`` action space
that the real wrapper exposes.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from invar_rl.layer3_control.phase3_env_wrappers import (
    CompactObservationWrapper,
    OnlineSharpeRewardWrapper,
)
from src.models.robust_invar_rl.compact_obs import (
    CompactObservationConfig,
    CompactObservationTape,
    N_BASE_FIELDS,
    N_REGIME_CLUSTERS,
)
from src.models.robust_invar_rl.online_sharpe_reward import (
    OnlineSharpeRewardConfig,
)


class _StubKellyPriorEnv(gym.Env):
    """Minimal stand-in for KellyPriorEnvWrapper."""

    metadata = {"render_modes": []}

    def __init__(self, T: int = 30, inner_obs_dim: int = 36) -> None:
        super().__init__()
        self._T = int(T)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(inner_obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.float32(-1.0), high=np.float32(1.0),
            shape=(1,), dtype=np.float32,
        )
        self._t = 0
        self._rng = np.random.default_rng(0)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        self._t = 0
        obs = np.zeros(
            self.observation_space.shape[0], dtype=np.float32
        )
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        a = float(np.asarray(action).reshape(-1)[0])
        r = float(self._rng.normal(0.0, 0.01))
        e_final = float(np.clip(0.8 + 0.25 * a, 0.0, 1.5))
        self._t += 1
        obs = np.zeros(
            self.observation_space.shape[0], dtype=np.float32
        )
        info: Dict = {
            "strategy_return": r,
            "exposure": e_final,
            "e_star": 0.8,
            "residual_action": a,
            "e_final": e_final,
            "equity": 1.0,
        }
        term = False
        trunc = self._t >= self._T - 1
        # Canonical reward placeholder so the OnlineSharpe wrapper can
        # stash it; the actual value is overridden.
        return obs, r, term, trunc, info


def _make_tape(T: int = 30) -> CompactObservationTape:
    rng = np.random.default_rng(0)
    ids = rng.integers(0, N_REGIME_CLUSTERS, size=T)
    oh = np.zeros((T, N_REGIME_CLUSTERS), dtype=np.float32)
    oh[np.arange(T), ids] = 1.0
    return CompactObservationTape(
        p_hat=rng.uniform(0.3, 0.7, size=T),
        mu_hat=rng.normal(0.0, 0.005, size=T),
        sigma_hat=rng.uniform(0.005, 0.02, size=T),
        e_star=rng.uniform(0.3, 1.5, size=T),
        vix_per_day=rng.uniform(10.0, 35.0, size=T),
        ust10y_per_day=rng.uniform(1.0, 4.5, size=T),
        regime_one_hot=oh,
    )


def test_compact_wrapper_obs_dim_is_17() -> None:
    stub = _StubKellyPriorEnv(T=20)
    tape = _make_tape(T=20)
    env = CompactObservationWrapper(stub, tape=tape)
    assert env.obs_dim == 17
    assert env.observation_space.shape == (17,)


def test_compact_wrapper_preserves_action_space() -> None:
    stub = _StubKellyPriorEnv(T=20)
    tape = _make_tape(T=20)
    env = CompactObservationWrapper(stub, tape=tape)
    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (1,)
    assert float(env.action_space.low[0]) == pytest.approx(-1.0)
    assert float(env.action_space.high[0]) == pytest.approx(+1.0)


def test_online_sharpe_wrapper_preserves_action_space() -> None:
    stub = _StubKellyPriorEnv(T=20)
    env = OnlineSharpeRewardWrapper(stub)
    assert env.action_space.shape == (1,)


def test_stacked_wrappers_preserve_action_space() -> None:
    stub = _StubKellyPriorEnv(T=20)
    tape = _make_tape(T=20)
    inner = CompactObservationWrapper(stub, tape=tape)
    outer = OnlineSharpeRewardWrapper(inner)
    assert outer.observation_space.shape == (17,)
    assert outer.action_space.shape == (1,)


def test_compact_wrapper_emits_correct_obs_shape_each_step() -> None:
    stub = _StubKellyPriorEnv(T=10)
    tape = _make_tape(T=10)
    env = CompactObservationWrapper(stub, tape=tape)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (17,)
    for _ in range(5):
        obs, _r, _term, _trunc, _info = env.step(np.asarray([0.0]))
        assert obs.shape == (17,)
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all()


def test_stacked_step_returns_finite_reward() -> None:
    stub = _StubKellyPriorEnv(T=15)
    tape = _make_tape(T=15)
    inner = CompactObservationWrapper(stub, tape=tape)
    outer = OnlineSharpeRewardWrapper(
        inner,
        cfg=OnlineSharpeRewardConfig(warmup_steps=3, clip=8.0),
    )
    obs, _ = outer.reset(seed=0)
    rews = []
    for _ in range(10):
        obs, r, term, trunc, info = outer.step(np.asarray([0.1]))
        assert np.isfinite(r)
        rews.append(r)
        assert "reward_canonical" in info
        assert "reward_online_sharpe" in info
    assert all(abs(r) <= 8.0 + 1e-6 for r in rews)


def test_compact_wrapper_reset_resets_builder() -> None:
    stub = _StubKellyPriorEnv(T=20)
    tape = _make_tape(T=20)
    env = CompactObservationWrapper(stub, tape=tape)
    obs0, _ = env.reset(seed=0)
    for _ in range(5):
        env.step(np.asarray([-0.5]))
    obs_reset, _ = env.reset(seed=1)
    # Drawdown / hit-rate / turnover counters should be zero again.
    assert float(obs_reset[3]) == pytest.approx(0.0)
    assert float(obs_reset[4]) == pytest.approx(0.0)
    assert float(obs_reset[5]) == pytest.approx(0.0)


def test_dim_without_regime_is_9() -> None:
    stub = _StubKellyPriorEnv(T=10)
    tape = _make_tape(T=10)
    env = CompactObservationWrapper(
        stub,
        tape=CompactObservationTape(
            p_hat=tape.p_hat,
            mu_hat=tape.mu_hat,
            sigma_hat=tape.sigma_hat,
            e_star=tape.e_star,
            vix_per_day=tape.vix_per_day,
            ust10y_per_day=tape.ust10y_per_day,
            regime_one_hot=None,
        ),
        cfg=CompactObservationConfig(include_regime_one_hot=True),
    )
    assert env.obs_dim == 9
    assert env.observation_space.shape == (9,)


def test_compact_obs_e_star_is_last_dim() -> None:
    stub = _StubKellyPriorEnv(T=10)
    tape = _make_tape(T=10)
    env = CompactObservationWrapper(stub, tape=tape)
    obs, _ = env.reset(seed=0)
    assert float(obs[-1]) == pytest.approx(float(tape.e_star[0]))


def test_obs_dim_constants_match_spec() -> None:
    assert N_BASE_FIELDS == 8
    assert N_REGIME_CLUSTERS == 8
