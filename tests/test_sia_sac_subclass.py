"""Construction + single-step training test for :class:`SACSIA`.

Requires Stable-Baselines3. Skipped on the local box if SB3 missing; runs
on Wulver where SB3 is part of the project venv.
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


def test_sacsia_constructs() -> None:
    from invar_rl.layer2_sia.sac_sia import SACSIA

    macro_dim = 16
    env = _make_env(7 + macro_dim)
    agent = SACSIA(
        policy="MlpPolicy", env=env,
        sia_config=SIAConfig(total_timesteps=10, buffer_size=200),
        macro_dim=macro_dim, l1_uncertainty=0,
        verbose=0, seed=42, device="cpu",
    )
    assert hasattr(agent, "sia_actor")
    assert hasattr(agent, "critic")
    assert hasattr(agent, "critic_target")


def test_sacsia_predict_returns_scalar_exposure() -> None:
    from invar_rl.layer2_sia.sac_sia import SACSIA

    macro_dim = 8
    env = _make_env(7 + macro_dim)
    agent = SACSIA(
        policy="MlpPolicy", env=env,
        sia_config=SIAConfig(total_timesteps=10, buffer_size=100),
        macro_dim=macro_dim, verbose=0, seed=7, device="cpu",
    )
    obs = np.random.randn(7 + macro_dim).astype(np.float32)
    action, _ = agent.predict(obs, deterministic=True)
    assert action.shape == (1,)
    assert 0.0 <= float(action[0]) <= 1.5 + 1e-6


def test_sacsia_single_training_step_does_not_crash() -> None:
    from invar_rl.layer2_sia.sac_sia import SACSIA

    macro_dim = 8
    env = _make_env(7 + macro_dim)
    agent = SACSIA(
        policy="MlpPolicy", env=env,
        sia_config=SIAConfig(total_timesteps=400, buffer_size=400),
        macro_dim=macro_dim, learning_starts=50,
        verbose=0, seed=11, device="cpu",
    )
    # learn for just enough steps that train() is invoked at least once.
    agent.learn(total_timesteps=200, progress_bar=False)
    stats = agent.sia_train_stats()
    # Aux + critic losses must be finite numbers; gates in [0, 1].
    assert np.isfinite(stats["critic_loss"])
    assert np.isfinite(stats["actor_loss"])
    assert np.isfinite(stats["aux_total"])
    for k in range(5):
        assert 0.0 <= stats[f"gate_{k}"] <= 1.0
