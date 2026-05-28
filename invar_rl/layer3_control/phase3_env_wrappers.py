"""Robust-InVAR-RL Phase 3: env wrappers for compact obs + online Sharpe.

This module exposes two gymnasium wrappers that compose on top of the
Phase 2 :class:`invar_rl.layer3_control.kelly_prior_env.KellyPriorEnvWrapper`
(which itself wraps :class:`invar_rl.layer3_control.env.ExposureEnv`):

- :class:`CompactObservationWrapper`: replaces the observation with a
  small set of sufficient statistics (see
  :class:`src.models.robust_invar_rl.compact_obs.CompactObservationBuilder`)
  while leaving the action space and reward unchanged.

- :class:`OnlineSharpeRewardWrapper`: replaces the per-step reward with
  an EWMA Sharpe-increment (see
  :class:`src.models.robust_invar_rl.online_sharpe_reward.OnlineSharpeReward`)
  while leaving the action space and observation unchanged.

Recommended stack (outer to inner)::

    OnlineSharpeRewardWrapper(
        CompactObservationWrapper(
            KellyPriorEnvWrapper(
                ExposureEnv(...)
            ),
            tape=...,
            cfg=...,
        )
    )

Both wrappers preserve the underlying 1-D ``Box([-1, +1])`` action
space because they only intercept obs / reward in ``step`` and the
post-reset ``obs``.

Neither wrapper performs any forecasting, training, or feature
engineering; they are pure post-step adapters that read a precomputed
``CompactObservationTape`` and a stateful EWMA Sharpe estimator.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from invar_rl.layer3_control.kelly_prior_env import KellyPriorEnvWrapper
from src.models.robust_invar_rl.compact_obs import (
    CompactObservationBuilder,
    CompactObservationConfig,
    CompactObservationTape,
)
from src.models.robust_invar_rl.online_sharpe_reward import (
    OnlineSharpeReward,
    OnlineSharpeRewardConfig,
)


_LOG_PREFIX = "[Phase3-EnvWrappers]"

# Generous symmetric obs bounds. The values come from rolling
# fractional returns + a few z-scored macros, all naturally bounded.
# We keep the box low/high large enough that even an outlier day won't
# clip; the wrapper raises on NaN/inf inside the builder if anything
# goes wrong upstream.
_OBS_LOW: float = -1.0e3
_OBS_HIGH: float = 1.0e3


class CompactObservationWrapper(gym.Wrapper):
    """Replace the Kelly-prior obs with the Phase 3 compact obs.

    The action and reward channels are forwarded unchanged from the
    underlying wrapper. The observation space is rewritten to a
    fixed-length ``Box``.

    Args:
        env: A :class:`KellyPriorEnvWrapper` instance.
        tape: Per-step precomputed inputs for the compact builder.
            Length must match the underlying tape's episode length.
        cfg: Compact observation config (toggles + scales). Defaults
            to spec defaults (regime one-hot on, vix/20, ust10y/4).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: KellyPriorEnvWrapper,
        tape: CompactObservationTape,
        cfg: Optional[CompactObservationConfig] = None,
    ) -> None:
        super().__init__(env)
        self._builder = CompactObservationBuilder(tape=tape, cfg=cfg)
        dim = int(self._builder.obs_dim)
        self.observation_space = spaces.Box(
            low=np.full(dim, _OBS_LOW, dtype=np.float32),
            high=np.full(dim, _OBS_HIGH, dtype=np.float32),
            dtype=np.float32,
        )
        # Action space is inherited (1-D Box([-1, +1])) from the inner
        # KellyPriorEnvWrapper; gym.Wrapper forwards it automatically.
        self._step_idx = 0
        self._n_obs_overrides = 0

    @property
    def builder(self) -> CompactObservationBuilder:
        return self._builder

    @property
    def obs_dim(self) -> int:
        return int(self._builder.obs_dim)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        _, info = self.env.reset(seed=seed, options=options)
        self._builder.reset()
        self._step_idx = 0
        obs = self._builder.build(step_idx=self._step_idx)
        self._n_obs_overrides += 1
        return obs.astype(np.float32, copy=False), dict(info)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        _, reward, term, trunc, info = self.env.step(action)
        # Update rolling counters using THIS step's outcomes.
        self._builder.update(
            step_idx=self._step_idx,
            strategy_return=float(info["strategy_return"]),
            exposure=float(info["e_final"]),
        )
        self._step_idx += 1
        obs = self._builder.build(step_idx=self._step_idx)
        self._n_obs_overrides += 1
        return (
            obs.astype(np.float32, copy=False),
            float(reward),
            bool(term),
            bool(trunc),
            dict(info),
        )


class OnlineSharpeRewardWrapper(gym.Wrapper):
    """Replace the per-step reward with EWMA online Sharpe increment.

    The observation channel is forwarded unchanged. The reward channel
    is overridden each step using
    :class:`src.models.robust_invar_rl.online_sharpe_reward.OnlineSharpeReward`
    over the realised strategy return.

    Args:
        env: The wrapped env (typically
            :class:`CompactObservationWrapper` over
            :class:`KellyPriorEnvWrapper`).
        cfg: Online Sharpe config (half-life, warm-up, clip). Defaults
            to spec defaults (half-life 21 days, 5 warm-up steps,
            clip +/- 8 Sharpe units).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: gym.Env,
        cfg: Optional[OnlineSharpeRewardConfig] = None,
    ) -> None:
        super().__init__(env)
        self._sharpe = OnlineSharpeReward(cfg=cfg)
        self._n_reward_overrides = 0

    @property
    def online_sharpe(self) -> OnlineSharpeReward:
        return self._sharpe

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._sharpe.reset()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, _raw_reward, term, trunc, info = self.env.step(action)
        r = float(info["strategy_return"])
        new_reward = self._sharpe.step(r)
        # Stash the canonical reward so anyone inspecting info can see
        # what was overridden.
        info = dict(info)
        info["reward_canonical"] = float(_raw_reward)
        info["reward_online_sharpe"] = float(new_reward)
        self._n_reward_overrides += 1
        return obs, float(new_reward), bool(term), bool(trunc), info


__all__ = [
    "CompactObservationWrapper",
    "OnlineSharpeRewardWrapper",
]
