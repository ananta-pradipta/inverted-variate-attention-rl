"""Robust-InVAR-RL Phase 2: Kelly-prior env wrapper for residual SAC.

The :class:`KellyPriorEnvWrapper` wraps the canonical
:class:`invar_rl.layer3_control.env.ExposureEnv` so that:

1. The observation grows by one dimension: a precomputed per-step
   Kelly-style prior ``e_star_t`` is appended as the last entry. The
   actor can therefore condition on the prior when learning its
   residual.

2. The 1-D action ``a in [-1, +1]`` is interpreted as a residual on
   top of ``e_star_t``::

       e_final = clip(e_star_t + delta_cap * a, 0, e_max)

   and ``e_final`` is then passed to the underlying ``ExposureEnv``
   for the rest of the step (clip-to-min/max, change-band, return
   computation, reward, etc.).

The wrapper preserves the SB3 SAC training loop completely: the
action space stays a 1-D ``Box([-1, +1])`` from the SAC perspective,
and the env-side decode is invisible to the optimiser. Stable-
Baselines3 SAC's MlpPolicy uses a ``tanh`` squashing function on the
actor output, which already gives the ``[-1, +1]`` interval the
wrapper expects; the only contract change is that the env now
interprets that interval as a residual rather than an absolute
exposure target.

The ``e_star`` tape is computed BEFORE training (see
:func:`src.models.robust_invar_rl.prior_exposure.build_e_star_tape`)
and supplied at wrap time. No live recomputation is performed in
``step``; the wrapper is a pure interpretation layer.
"""
from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from invar_rl.layer3_control.env import ExposureEnv

_LOG_PREFIX = "[Phase2-KellyPriorEnv]"


class KellyPriorEnvWrapper(gym.Env):
    """Env wrapper that turns the SAC action into a Kelly-prior residual.

    Attributes:
        delta_cap: Max absolute residual the actor can add to ``e_star``
            (in exposure units). E.g. ``delta_cap=0.25`` means the
            actor can bend exposure by at most ``+/- 0.25`` around the
            prior, regardless of where the prior sits inside ``[0, e_max]``.
        e_max: Hard upper bound on the final exposure (matches the
            underlying ``Layer3Config.exposure_max`` so the wrapper
            never tries to exceed the env's own clip).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        inner_env: ExposureEnv,
        e_star_tape: np.ndarray,
        delta_cap: float,
        e_max: float,
        e_min: float = 0.0,
    ) -> None:
        super().__init__()
        if delta_cap <= 0.0:
            raise ValueError(
                f"[ERR] delta_cap must be > 0; got {delta_cap}"
            )
        if e_max <= 0.0:
            raise ValueError(
                f"[ERR] e_max must be > 0; got {e_max}"
            )
        if e_min < 0.0 or e_min >= e_max:
            raise ValueError(
                f"[ERR] e_min must be in [0, e_max); got {e_min} vs {e_max}"
            )
        e_star_tape = np.asarray(e_star_tape, dtype=np.float64).ravel()
        if e_star_tape.size < 2:
            raise ValueError(
                f"[ERR] e_star_tape must have >= 2 entries; got {e_star_tape.size}"
            )
        if not np.isfinite(e_star_tape).all():
            raise ValueError("[ERR] e_star_tape contains NaN or inf")
        if (e_star_tape < 0.0).any() or (e_star_tape > e_max + 1.0e-9).any():
            raise ValueError(
                "[ERR] e_star_tape values must lie in [0, e_max]; "
                f"got min={e_star_tape.min()} max={e_star_tape.max()} e_max={e_max}"
            )

        self._inner = inner_env
        self._e_star = e_star_tape
        self._delta_cap = float(delta_cap)
        self._e_max = float(e_max)
        self._e_min = float(e_min)
        self._step_idx = 0

        # Observation = inner-env observation plus a single scalar (e_star_t).
        inner_obs_space = inner_env.observation_space
        if not isinstance(inner_obs_space, spaces.Box):
            raise TypeError(
                "[ERR] inner_env.observation_space must be a Box; got "
                f"{type(inner_obs_space).__name__}"
            )
        inner_low = inner_obs_space.low.astype(np.float32)
        inner_high = inner_obs_space.high.astype(np.float32)
        new_low = np.concatenate(
            [inner_low, np.asarray([0.0], dtype=np.float32)]
        )
        new_high = np.concatenate(
            [inner_high, np.asarray([self._e_max], dtype=np.float32)]
        )
        self.observation_space = spaces.Box(
            low=new_low, high=new_high, dtype=np.float32
        )

        # Action space is a 1-D residual in [-1, +1]. SAC's tanh-squashed
        # actor naturally produces values in this range.
        self.action_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(1,),
            dtype=np.float32,
        )

    @property
    def delta_cap(self) -> float:
        return self._delta_cap

    @property
    def e_max(self) -> float:
        return self._e_max

    def _e_star_at(self, idx: int) -> float:
        T = self._e_star.shape[0]
        # Clamp to tape range; the env's truncation logic handles end-of-tape.
        return float(self._e_star[int(max(0, min(T - 1, idx)))])

    def _append_e_star(self, obs: np.ndarray, idx: int) -> np.ndarray:
        obs32 = np.asarray(obs, dtype=np.float32).ravel()
        return np.concatenate(
            [obs32, np.asarray([self._e_star_at(idx)], dtype=np.float32)]
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        inner_obs, info = self._inner.reset(seed=seed, options=options)
        # Inner env's _start is the offset into the tape; we mirror it
        # locally for our own e_star lookup.
        start = int(getattr(self._inner, "_start", 0))
        self._step_idx = start
        return self._append_e_star(inner_obs, self._step_idx), dict(info)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Decode residual action: SAC supplies a in [-1, +1] (tanh-squashed
        # under MlpPolicy). e_star is the per-step prior; the residual is
        # bounded by delta_cap; final exposure is clipped to [e_min, e_max].
        a = float(np.asarray(action).reshape(-1)[0])
        a = float(np.clip(a, -1.0, 1.0))
        e_star_t = self._e_star_at(self._step_idx)
        residual = self._delta_cap * a
        e_final = float(
            np.clip(e_star_t + residual, self._e_min, self._e_max)
        )
        # Pass the decoded final exposure to the inner env. The inner
        # env's own action_space allows the full [exposure_min,
        # exposure_max] band, so we just submit e_final as a 1-D array.
        inner_action = np.asarray([e_final], dtype=np.float32)
        inner_obs, reward, term, trunc, info = self._inner.step(
            inner_action
        )
        info = dict(info)
        info["e_star"] = float(e_star_t)
        info["residual_action"] = float(a)
        info["e_final"] = float(e_final)
        self._step_idx += 1
        obs = self._append_e_star(inner_obs, self._step_idx)
        return obs, float(reward), bool(term), bool(trunc), info

    def render(self) -> None:
        # Render delegated to inner env (no-op for ExposureEnv).
        return None

    def close(self) -> None:
        if hasattr(self._inner, "close"):
            try:
                self._inner.close()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"{_LOG_PREFIX} [WARN] inner env close raised: {exc}"
                )


__all__ = ["KellyPriorEnvWrapper"]
