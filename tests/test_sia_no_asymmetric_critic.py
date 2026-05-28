"""Tests for the Phase 4 ``no_a`` ablation (SACSIA asymmetric_critic=False).

With ``asymmetric_critic=False`` the SB3 twin-Q critic is rebuilt on the
actor's post-gate bottleneck ``actor_in`` (1 + 2 + macro_small + 4 + 1)
instead of the full observation. Requires Stable-Baselines3.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

sb3 = importlib.util.find_spec("stable_baselines3")
if sb3 is None:  # pragma: no cover
    pytest.skip("stable_baselines3 not installed", allow_module_level=True)


from invar_rl.layer2_sia.config import SIAConfig


def _make_env(obs_dim: int):
    import gymnasium as gym

    class TinyEnv(gym.Env):
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
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
            return np.zeros(obs_dim, dtype=np.float32), {}

        def step(self, action):
            self._t += 1
            obs = np.random.randn(obs_dim).astype(np.float32)
            r = float(np.random.randn() * 0.01)
            term = bool(self._t >= self._n)
            return obs, r, term, False, {}

    return TinyEnv()


def test_no_asymmetric_critic_critic_input_dim_matches_actor_in() -> None:
    """Critic was rebuilt on actor_in_dim, not full obs_dim."""
    from invar_rl.layer2_sia.sac_sia import SACSIA

    macro_dim = 16
    env = _make_env(7 + macro_dim)
    agent = SACSIA(
        policy="MlpPolicy", env=env,
        sia_config=SIAConfig(
            total_timesteps=10, buffer_size=200,
            asymmetric_critic=False,
        ),
        macro_dim=macro_dim, l1_uncertainty=0,
        verbose=0, seed=42, device="cpu",
    )
    actor_in_dim = int(agent.sia_actor.actor_in_dim)
    # Expected: 1 (disp) + 2 (wrapper) + macro_small_dim (16) + 4 (risk) + 1 (l1u)
    assert actor_in_dim == 1 + 2 + 16 + 4 + 1
    # The critic's observation_space must now report actor_in_dim, not the
    # full obs_dim (= 7 + 16 = 23).
    critic_obs_shape = agent.critic.observation_space.shape
    assert critic_obs_shape == (actor_in_dim,)
    assert critic_obs_shape != (7 + macro_dim,)
    # Same for the Polyak target.
    target_obs_shape = agent.critic_target.observation_space.shape
    assert target_obs_shape == (actor_in_dim,)


def test_asymmetric_critic_default_keeps_full_obs() -> None:
    """Default (asymmetric_critic=True) keeps the critic on the full obs."""
    from invar_rl.layer2_sia.sac_sia import SACSIA

    macro_dim = 8
    env = _make_env(7 + macro_dim)
    agent = SACSIA(
        policy="MlpPolicy", env=env,
        sia_config=SIAConfig(total_timesteps=10, buffer_size=100),
        macro_dim=macro_dim, verbose=0, seed=7, device="cpu",
    )
    full_obs_dim = 7 + macro_dim
    assert agent.critic.observation_space.shape == (full_obs_dim,)
    assert agent.critic_target.observation_space.shape == (full_obs_dim,)


def test_no_asymmetric_critic_single_training_step_does_not_crash() -> None:
    """Critic forwards on the bottleneck must accept the smaller obs."""
    from invar_rl.layer2_sia.sac_sia import SACSIA

    macro_dim = 8
    env = _make_env(7 + macro_dim)
    agent = SACSIA(
        policy="MlpPolicy", env=env,
        sia_config=SIAConfig(
            total_timesteps=400, buffer_size=400,
            asymmetric_critic=False,
        ),
        macro_dim=macro_dim, learning_starts=50,
        verbose=0, seed=11, device="cpu",
    )
    agent.learn(total_timesteps=200, progress_bar=False)
    stats = agent.sia_train_stats()
    assert np.isfinite(stats["critic_loss"])
    assert np.isfinite(stats["actor_loss"])
    assert np.isfinite(stats["aux_total"])
