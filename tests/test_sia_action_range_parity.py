"""Action-range parity test for SACSIA (fix verification for audit bug B1).

The canonical SB3 SAC contract is:

- The env's action space is ``Box(low=0, high=1.5)``.
- ``predict()`` returns UNSCALED actions in ``[0, 1.5]``; the env step
  consumes them in that range.
- SB3's ``_sample_action`` then runs ``scale_action`` to map back to
  ``[-1, 1]`` for replay-buffer storage; every replay-buffer transition
  therefore carries an action in ``[-1, 1]``.
- The critic's per-step forward and per-step target forward both see
  actions in ``[-1, 1]``: the actor's tanh-squashed output for ``pi(s)``
  and the buffer-stored action for the current Q.

Pre-fix SACSIA broke this invariant by emitting actions in ``[0, 1.5]``
from ``predict()`` (so the buffer stored ``[-1, 1]`` scaled versions
correctly via SB3), but the train() loop ALSO used unscaled ``[0, 1.5]``
actions for the next-state Q target. That mismatch is fixed in this
audit by routing the SIA actor through the SB3 squashed-Gaussian
contract.

These tests assert the post-fix invariants on a synthetic env.
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


def _make_env(obs_dim: int = 23, action_low: float = 0.0,
              action_high: float = 1.5):
    """Tiny synthetic env with Box action space [action_low, action_high]."""
    import gymnasium as gym

    class TinyEnv(gym.Env):
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )
        action_space = gym.spaces.Box(
            low=np.float32(action_low), high=np.float32(action_high),
            shape=(1,), dtype=np.float32,
        )

        def __init__(self) -> None:
            super().__init__()
            self._t = 0
            self._n = 64
            self.last_step_action = None

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._t = 0
            return np.zeros(obs_dim, dtype=np.float32), {}

        def step(self, action):
            # Record the action handed to the env so the test can verify
            # it lives in [action_low, action_high].
            self.last_step_action = np.asarray(action, dtype=np.float32).copy()
            self._t += 1
            obs = np.random.randn(obs_dim).astype(np.float32)
            r = float(np.random.randn() * 0.01)
            term = bool(self._t >= self._n)
            return obs, r, term, False, {}

    return TinyEnv()


def test_predict_action_is_in_env_range() -> None:
    """``agent.predict()`` returns actions in [exposure_low, exposure_high]."""
    from invar_rl.layer2_sia.sac_sia import SACSIA

    env = _make_env(obs_dim=23, action_low=0.0, action_high=1.5)
    agent = SACSIA(
        policy="MlpPolicy", env=env,
        sia_config=SIAConfig(total_timesteps=10, buffer_size=200),
        macro_dim=16, l1_uncertainty=0,
        verbose=0, seed=42, device="cpu",
    )
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs = rng.standard_normal(23).astype(np.float32)
        action_det, _ = agent.predict(obs, deterministic=True)
        action_sto, _ = agent.predict(obs, deterministic=False)
        assert action_det.shape == (1,)
        assert action_sto.shape == (1,)
        for a in (action_det, action_sto):
            assert 0.0 - 1e-6 <= float(a[0]) <= 1.5 + 1e-6, (
                f"predict() returned action {a[0]} outside [0, 1.5]"
            )


def test_buffer_stores_actions_in_minus_one_to_one() -> None:
    """SB3's ``_sample_action`` scales env actions back to [-1, 1] storage.

    After ``learn()`` fills the replay buffer, every stored action must
    live in [-1, 1] regardless of the env's action range. This is the
    invariant the critic's per-step forward relies on.
    """
    from invar_rl.layer2_sia.sac_sia import SACSIA

    env = _make_env(obs_dim=23, action_low=0.0, action_high=1.5)
    agent = SACSIA(
        policy="MlpPolicy", env=env,
        sia_config=SIAConfig(total_timesteps=400, buffer_size=400),
        macro_dim=16, learning_starts=50,
        verbose=0, seed=11, device="cpu",
    )
    agent.learn(total_timesteps=200, progress_bar=False)
    # Slice the populated portion of the buffer.
    n = int(agent.replay_buffer.size())
    assert n > 0, "replay buffer empty after learn()"
    stored = agent.replay_buffer.actions[:n]
    lo, hi = float(stored.min()), float(stored.max())
    assert lo >= -1.0 - 1e-6, (
        f"buffer action min {lo} < -1 (action-space scaling broken)"
    )
    assert hi <= 1.0 + 1e-6, (
        f"buffer action max {hi} > 1 (action-space scaling broken)"
    )


def test_env_step_receives_actions_in_unscaled_range() -> None:
    """SB3 calls ``env.step(action)`` with un-scaled actions in [low, high].

    After learn(), the env's ``last_step_action`` (the most recent action
    passed to env.step) must live in the env's declared action space.
    """
    from invar_rl.layer2_sia.sac_sia import SACSIA

    env = _make_env(obs_dim=23, action_low=0.0, action_high=1.5)
    agent = SACSIA(
        policy="MlpPolicy", env=env,
        sia_config=SIAConfig(total_timesteps=400, buffer_size=400),
        macro_dim=16, learning_starts=50,
        verbose=0, seed=23, device="cpu",
    )
    agent.learn(total_timesteps=200, progress_bar=False)
    # SB3 wraps env in a DummyVecEnv; access the underlying env's record.
    # The agent's training-time env is the wrapped vector env; the
    # original env we built is preserved as env.envs[0].
    inner_env = agent.env.envs[0]
    last = inner_env.unwrapped.last_step_action
    if last is None:
        pytest.skip("env never received a step (learn() did not run train)")
    assert 0.0 - 1e-6 <= float(last.reshape(-1)[0]) <= 1.5 + 1e-6, (
        f"env.step received action {last} outside [0, 1.5]"
    )


def test_critic_consistent_action_axis_in_train() -> None:
    """In ``train()``, both critic Q calls use scaled actions in [-1, 1].

    We assert the invariants directly by patching ``critic.forward`` to
    record the action ranges it sees during a single train() step and
    verifying both current-Q (buffer action) and target-Q (next_action
    from the SIA actor) live in [-1, 1].
    """
    from invar_rl.layer2_sia.sac_sia import SACSIA

    env = _make_env(obs_dim=23, action_low=0.0, action_high=1.5)
    agent = SACSIA(
        policy="MlpPolicy", env=env,
        sia_config=SIAConfig(total_timesteps=400, buffer_size=400),
        macro_dim=16, learning_starts=50,
        verbose=0, seed=29, device="cpu",
    )
    # Fill replay buffer with one short learn().
    agent.learn(total_timesteps=120, progress_bar=False)

    seen_action_ranges = []

    real_critic_forward = agent.critic.forward
    real_critic_target_forward = agent.critic_target.forward

    def _spy_critic(obs, action):
        seen_action_ranges.append(
            ("critic", float(action.min().item()), float(action.max().item()))
        )
        return real_critic_forward(obs, action)

    def _spy_critic_target(obs, action):
        seen_action_ranges.append(
            ("critic_target",
             float(action.min().item()), float(action.max().item()))
        )
        return real_critic_target_forward(obs, action)

    agent.critic.forward = _spy_critic
    agent.critic_target.forward = _spy_critic_target
    try:
        agent.train(gradient_steps=1, batch_size=32)
    finally:
        agent.critic.forward = real_critic_forward
        agent.critic_target.forward = real_critic_target_forward

    assert len(seen_action_ranges) >= 2, (
        f"train() did not exercise both critics (saw {seen_action_ranges})"
    )
    for name, lo, hi in seen_action_ranges:
        assert lo >= -1.0 - 1e-6 and hi <= 1.0 + 1e-6, (
            f"{name} received action range [{lo}, {hi}] outside [-1, 1]"
        )
