"""Tests for the Phase 2 KellyPriorEnvWrapper.

These tests stub the inner :class:`ExposureEnv` with a lightweight
gymnasium env that only validates the wrapper's contract: that the
observation grows by one dim, the action is decoded as a residual
over ``e_star_t``, and the final exposure submitted to the inner env
is clipped to ``[e_min, e_max]``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from invar_rl.layer3_control.kelly_prior_env import KellyPriorEnvWrapper


class _StubInnerEnv(gym.Env):
    """Stub stand-in for ExposureEnv used by the wrapper tests."""

    metadata = {"render_modes": []}

    def __init__(self, T: int = 10, obs_dim: int = 4) -> None:
        super().__init__()
        self._T = int(T)
        self._t = 0
        self._start = 0
        self.received_actions: list = []
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.float32(0.0), high=np.float32(1.5),
            shape=(1,), dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        self._t = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        a = float(np.asarray(action).reshape(-1)[0])
        self.received_actions.append(a)
        self._t += 1
        obs = np.full(
            self.observation_space.shape, fill_value=float(self._t),
            dtype=np.float32,
        )
        info = {"equity": 1.0, "exposure": a, "strategy_return": 0.01 * a}
        return obs, 0.0, False, self._t >= self._T - 1, info


def test_observation_dim_grows_by_one() -> None:
    inner = _StubInnerEnv(obs_dim=4)
    tape = np.linspace(0.0, 1.0, 10)
    w = KellyPriorEnvWrapper(
        inner_env=inner, e_star_tape=tape,
        delta_cap=0.25, e_max=1.5,
    )
    assert w.observation_space.shape == (5,)
    obs, _ = w.reset()
    assert obs.shape == (5,)
    assert obs[-1] == pytest.approx(tape[0])


def test_residual_decode_matches_spec() -> None:
    inner = _StubInnerEnv()
    tape = np.full(20, 0.5)
    w = KellyPriorEnvWrapper(
        inner_env=inner, e_star_tape=tape,
        delta_cap=0.25, e_max=1.5,
    )
    w.reset()
    # action = +1.0 => e_final = clip(0.5 + 0.25 * 1.0, 0, 1.5) = 0.75
    _, _, _, _, info = w.step(np.asarray([1.0], dtype=np.float32))
    assert info["e_final"] == pytest.approx(0.75)
    assert inner.received_actions[-1] == pytest.approx(0.75)
    # action = -1.0 => e_final = clip(0.5 - 0.25, 0, 1.5) = 0.25
    _, _, _, _, info = w.step(np.asarray([-1.0], dtype=np.float32))
    assert info["e_final"] == pytest.approx(0.25)


def test_action_clipped_to_unit() -> None:
    inner = _StubInnerEnv()
    tape = np.full(20, 0.5)
    w = KellyPriorEnvWrapper(
        inner_env=inner, e_star_tape=tape,
        delta_cap=0.25, e_max=1.5,
    )
    w.reset()
    # action +5 should clip to +1 then decode to 0.75
    _, _, _, _, info = w.step(np.asarray([5.0], dtype=np.float32))
    assert info["e_final"] == pytest.approx(0.75)
    assert info["residual_action"] == pytest.approx(1.0)


def test_final_exposure_within_bounds() -> None:
    inner = _StubInnerEnv()
    # e_star always 1.4, with delta_cap 0.25 and e_max 1.5 => max e_final =
    # clip(1.65, 0, 1.5) = 1.5
    tape = np.full(20, 1.4)
    w = KellyPriorEnvWrapper(
        inner_env=inner, e_star_tape=tape,
        delta_cap=0.25, e_max=1.5,
    )
    w.reset()
    _, _, _, _, info = w.step(np.asarray([1.0], dtype=np.float32))
    assert info["e_final"] == pytest.approx(1.5)


def test_invalid_tape_values_rejected() -> None:
    inner = _StubInnerEnv()
    bad_tape = np.asarray([0.5, 2.0, 0.3])  # 2.0 > e_max
    with pytest.raises(ValueError):
        KellyPriorEnvWrapper(
            inner_env=inner, e_star_tape=bad_tape,
            delta_cap=0.25, e_max=1.5,
        )


def test_delta_cap_must_be_positive() -> None:
    inner = _StubInnerEnv()
    tape = np.full(5, 0.5)
    with pytest.raises(ValueError):
        KellyPriorEnvWrapper(
            inner_env=inner, e_star_tape=tape,
            delta_cap=0.0, e_max=1.5,
        )
