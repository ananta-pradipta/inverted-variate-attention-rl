"""Env wrapper that appends a per-day k-means-8 regime cluster id to obs.

The Sparse Invariant Actor's auxiliary loss includes a regime-invariance
term that needs a per-transition group id. SB3's replay buffer stores
``(obs, action, reward, next_obs, done)`` only; the day index is not
preserved. The cleanest way to carry the group id through SB3's pipeline
without surgery on the replay buffer is to extend the observation by one
trailing dim that holds ``float(cluster_id)`` for the day at step ``t``.

The Sparse Invariant Actor strips this trailing dim from its gate and
encoder inputs (see :class:`~invar_rl.layer2_sia.sparse_actor.SparseInvariantActor`),
so the cluster id never reaches the actor's forward. The SAC twin-Q
critics do see the full obs (including the trailing dim), but the critic
gradient through a discrete 0..7 column is effectively a small bias and
does not break the SAC design (the critic already conditions on macro
state via the canonical macro_encoding block).

Usage::

    from invar_rl.layer2_sia.regime_probs import load_probs_lookup
    lookup = load_probs_lookup("sp500", fold=1)
    # build per-day argmax cluster id from the soft-probs lookup
    day_to_cluster = {
        int(d): int(np.argmax(lookup[int(d)])) for d in lookup
    }
    wrapped = RegimeLabelEnv(env, tape_days=tape.days, day_to_cluster=day_to_cluster)
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RegimeLabelEnv(gym.Wrapper):
    """Append a trailing regime cluster id (float 0..7) to the observation.

    The wrapped env's ``observation_space`` is widened by 1 dim. The
    cluster id at step ``t`` is looked up from ``tape_days[start + t]``;
    if the day index is missing from ``day_to_cluster``, the wrapper
    falls back to ``fallback_cluster`` (default 0).

    The wrapper does not modify reward, termination, or the underlying
    env's action_space. It does not assume the wrapped env uses
    ``ExposureEnv``-style internals beyond exposing a ``_tape.days``
    attribute or a ``_start`` + ``_t`` step counter; instead it tracks
    its own counter against the per-reset starting day.

    Attributes:
        n_clusters: Number of clusters (purely for diagnostic use).
    """

    n_clusters: int = 8

    def __init__(
        self,
        env: gym.Env,
        tape_days: np.ndarray,
        day_to_cluster: Mapping[int, int],
        fallback_cluster: int = 0,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(
                "RegimeLabelEnv requires a Box observation_space; got "
                f"{type(env.observation_space).__name__}"
            )
        if env.observation_space.shape is None or len(env.observation_space.shape) != 1:
            raise ValueError(
                "RegimeLabelEnv requires a 1-D observation_space; got "
                f"shape {env.observation_space.shape}"
            )
        self._tape_days = np.asarray(tape_days, dtype=np.int64)
        self._day_to_cluster: Dict[int, int] = {
            int(k): int(v) for k, v in day_to_cluster.items()
        }
        self._fallback = int(fallback_cluster)

        inner = env.observation_space
        low = np.concatenate(
            [
                inner.low.astype(np.float32),
                np.array([0.0], dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                inner.high.astype(np.float32),
                np.array([float(self.n_clusters)], dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32,
        )
        self._step_idx: int = 0

    # ------------------------------------------------------------------
    def _cluster_for_step(self, step_idx: int) -> int:
        if step_idx < 0 or step_idx >= self._tape_days.shape[0]:
            return self._fallback
        day = int(self._tape_days[step_idx])
        return int(self._day_to_cluster.get(day, self._fallback))

    def _append_label(
        self, obs: np.ndarray, step_idx: int
    ) -> np.ndarray:
        cid = self._cluster_for_step(step_idx)
        tail = np.array([float(cid)], dtype=np.float32)
        return np.concatenate(
            [np.asarray(obs, dtype=np.float32), tail], axis=0
        )

    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        # Track the wrapped env's start position so the cluster id matches
        # the same day the inner env is about to act on.
        self._step_idx = int(getattr(self.env, "_start", 0))
        return self._append_label(obs, self._step_idx), info

    def step(
        self, action
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        # After step, the inner env's t advanced by 1.
        inner_t = int(getattr(self.env, "_t", 0))
        inner_start = int(getattr(self.env, "_start", 0))
        self._step_idx = min(
            inner_start + inner_t, self._tape_days.shape[0] - 1
        )
        return (
            self._append_label(obs, self._step_idx),
            reward, terminated, truncated, info,
        )


__all__ = ["RegimeLabelEnv"]
