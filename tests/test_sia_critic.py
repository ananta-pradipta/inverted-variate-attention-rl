"""Tests for the SIA full-info critic.

SIA reuses SB3 SAC's twin-Q critic on the full observation; we therefore
test the critic via a SACSIA construction and a single forward pass.
Requires Stable-Baselines3; the SACSIA construction is the canonical
entry point so we exercise the same wiring the driver uses.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

sb3 = importlib.util.find_spec("stable_baselines3")
if sb3 is None:  # pragma: no cover - skip path only
    pytest.skip("stable_baselines3 not installed", allow_module_level=True)


from invar_rl.layer2_sia.config import SIAConfig
from invar_rl.layer2_sia.sac_sia import SACSIA


def _make_env(obs_dim: int = 23):
    """Tiny synthetic Box env matching the SIA observation layout.

    obs_dim defaults to 23 = 7 (fixed) + 16 (macro) + 0 (l1u), matching
    the canonical SP500 macro_dim.
    """
    import gymnasium as gym

    class TinyEnv(gym.Env):
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )
        action_space = gym.spaces.Box(
            low=np.float32(0.0), high=np.float32(1.5),
            shape=(1,), dtype=np.float32,
        )

        def __init__(self) -> None:
            super().__init__()
            self._t = 0
            self._n = 64

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._t = 0
            return (
                np.zeros(obs_dim, dtype=np.float32),
                {},
            )

        def step(self, action):
            self._t += 1
            obs = np.random.randn(obs_dim).astype(np.float32)
            r = float(np.random.randn() * 0.01)
            term = bool(self._t >= self._n)
            return obs, r, term, False, {}

    return TinyEnv()


def test_critic_is_twin_q_on_full_obs() -> None:
    env = _make_env(obs_dim=23)
    agent = SACSIA(
        policy="MlpPolicy", env=env,
        sia_config=SIAConfig(total_timesteps=10),
        macro_dim=16, l1_uncertainty=0,
        verbose=0, seed=42, device="cpu",
    )
    # The SB3 critic returns a 2-tuple of (B, 1) tensors; the full obs is
    # what we feed (no stripping or gating).
    obs = torch.randn(5, 23)
    action = torch.rand(5, 1) * 1.5
    qs = agent.critic(obs, action)
    assert isinstance(qs, tuple)
    assert len(qs) == 2
    for q in qs:
        assert q.shape == (5, 1)


def test_critic_target_constructed_and_polyak_compatible() -> None:
    env = _make_env(obs_dim=23)
    agent = SACSIA(
        policy="MlpPolicy", env=env,
        sia_config=SIAConfig(total_timesteps=10),
        macro_dim=16, verbose=0, seed=42, device="cpu",
    )
    # Twin Q target exists and matches param shapes.
    target = agent.critic_target
    for p, p_t in zip(agent.critic.parameters(), target.parameters()):
        assert p.shape == p_t.shape
