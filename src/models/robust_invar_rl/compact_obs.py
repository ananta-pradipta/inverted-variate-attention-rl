"""Robust-InVAR-RL Phase 3: compact observation builder.

The canonical Layer 3 observation (``invar_rl/layer3_control/observation.py``)
packs 7 fixed fields + the full ``macro_encoding`` tail (z-scored macro
duration features, ~28 dims for the SP500 lattice_native panel). The
Phase 2 :class:`KellyPriorEnvWrapper` appends one more scalar
(``e_star_t``). On the SP500 panel that yields an observation of
roughly 7 + 28 + 1 = 36 dims.

Per the source design doc (2026-05-26), this width is hypothesised to
cause SAC over-control on small / mid universes because the actor is
asked to map a high-dimensional, partly redundant feature vector to a
1-D residual exposure. The compact obs replaces it with the smallest
set of sufficient statistics that the residual SAC needs:

    [
      0  p_hat_t            calibrated profitable-wrapper probability
      1  mu_hat_t            expected daily portfolio return (prior)
      2  sigma_hat_t         EWMA volatility (21-day half-life)
      3  drawdown_t          rolling 21-day drawdown of strategy NAV
      4  turnover_t          rolling 5-day mean turnover
      5  hit_rate_t          rolling 21-day fraction of profitable days
      6  vix_normalised_t    VIX scaled (raw / 20)
      7  ust10y_normalised_t UST10Y scaled (raw / 4)
    ]

Plus an optional regime-id one-hot (8 dims, from k-means-8).

Plus, ALWAYS APPENDED LAST, the per-step Kelly prior ``e_star_t`` (the
same scalar the :class:`KellyPriorEnvWrapper` exposes in its info dict).
This mirrors the wrapper's append-e_star convention so the actor still
sees the prior its action is residualising against.

Spec totals:
- 8 base + 8 regime + 1 e_star = 17 (default, all toggles on)
- 8 base + 1 e_star = 9 (regime off)

The builder is stateful: it owns per-step rolling counters for
drawdown, turnover, and hit-rate, computed from the wrapper's
realised PnL and exposure stream. The driver is responsible for
calling :meth:`reset` once per episode and :meth:`update` once per
step BEFORE :meth:`build` is asked for the new obs.

All values are float32. The full obs is bounded in spirit (NaN/inf
raises); the gym.Box low/high are kept generous because Phase 2's
inner ExposureEnv already publishes ``-inf, +inf`` for the canonical
state and SB3's MlpPolicy normalisation is identity on raw obs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np


_LOG_PREFIX = "[Phase3-CompactObs]"

# Base feature count (toggles + e_star are tracked separately).
N_BASE_FIELDS: int = 8

# Regime one-hot fixed at k-means-8.
N_REGIME_CLUSTERS: int = 8

# Default rolling-window lengths.
DRAWDOWN_WINDOW_DAYS: int = 21
TURNOVER_WINDOW_DAYS: int = 5
HIT_RATE_WINDOW_DAYS: int = 21


@dataclass
class CompactObservationConfig:
    """Hyperparameters for :class:`CompactObservationBuilder`."""

    include_regime_one_hot: bool = True
    vix_scale: float = 20.0
    ust10y_scale: float = 4.0
    drawdown_window: int = DRAWDOWN_WINDOW_DAYS
    turnover_window: int = TURNOVER_WINDOW_DAYS
    hit_rate_window: int = HIT_RATE_WINDOW_DAYS

    def __post_init__(self) -> None:
        if self.vix_scale <= 0.0:
            raise ValueError(
                f"[ERR] vix_scale must be > 0; got {self.vix_scale}"
            )
        if self.ust10y_scale <= 0.0:
            raise ValueError(
                f"[ERR] ust10y_scale must be > 0; got {self.ust10y_scale}"
            )
        if self.drawdown_window < 2:
            raise ValueError(
                "[ERR] drawdown_window must be >= 2; "
                f"got {self.drawdown_window}"
            )
        if self.turnover_window < 1:
            raise ValueError(
                "[ERR] turnover_window must be >= 1; "
                f"got {self.turnover_window}"
            )
        if self.hit_rate_window < 1:
            raise ValueError(
                "[ERR] hit_rate_window must be >= 1; "
                f"got {self.hit_rate_window}"
            )


@dataclass
class CompactObservationTape:
    """Per-step precomputed inputs for the builder.

    All arrays have length ``T`` (the wrapper-tape episode length).
    ``vix_per_day`` and ``ust10y_per_day`` are RAW (not z-scored) so
    the builder can apply the spec normalisation directly. ``p_hat``,
    ``mu_hat``, ``sigma_hat``, ``e_star`` come from the Phase 2 prior
    pipeline. ``regime_one_hot`` is optional (shape ``(T, K)``);
    if ``None``, the regime block is omitted regardless of the config
    toggle.
    """

    p_hat: np.ndarray
    mu_hat: np.ndarray
    sigma_hat: np.ndarray
    e_star: np.ndarray
    vix_per_day: np.ndarray
    ust10y_per_day: np.ndarray
    regime_one_hot: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        T = int(self.p_hat.shape[0])
        for name, arr in (
            ("p_hat", self.p_hat),
            ("mu_hat", self.mu_hat),
            ("sigma_hat", self.sigma_hat),
            ("e_star", self.e_star),
            ("vix_per_day", self.vix_per_day),
            ("ust10y_per_day", self.ust10y_per_day),
        ):
            if arr.shape[0] != T:
                raise ValueError(
                    f"[ERR] CompactObservationTape.{name} length "
                    f"mismatch: {arr.shape[0]} vs {T}"
                )
            if not np.isfinite(arr).all():
                raise ValueError(
                    f"[ERR] CompactObservationTape.{name} contains NaN/inf"
                )
        if self.regime_one_hot is not None:
            if self.regime_one_hot.shape[0] != T:
                raise ValueError(
                    "[ERR] CompactObservationTape.regime_one_hot length "
                    f"mismatch: {self.regime_one_hot.shape[0]} vs {T}"
                )
            if self.regime_one_hot.shape[1] != N_REGIME_CLUSTERS:
                raise ValueError(
                    "[ERR] CompactObservationTape.regime_one_hot must have "
                    f"{N_REGIME_CLUSTERS} columns; got "
                    f"{self.regime_one_hot.shape[1]}"
                )
            if not np.isfinite(self.regime_one_hot).all():
                raise ValueError(
                    "[ERR] CompactObservationTape.regime_one_hot contains "
                    "NaN/inf"
                )

    def __len__(self) -> int:
        return int(self.p_hat.shape[0])


@dataclass
class _RollingState:
    """Mutable per-episode state held by the builder."""

    equity: float = 1.0
    hwm: float = 1.0
    prev_exposure: float = 0.0
    return_hist: Deque[float] = field(default_factory=deque)
    turnover_hist: Deque[float] = field(default_factory=deque)
    hit_hist: Deque[int] = field(default_factory=deque)


class CompactObservationBuilder:
    """Stateful builder for the Phase 3 compact observation.

    Usage (per episode)::

        builder = CompactObservationBuilder(tape, cfg)
        builder.reset()
        obs_0 = builder.build(step_idx=0)            # initial obs
        for t in range(T):
            ...                                       # env.step(action)
            builder.update(
                step_idx=t,
                strategy_return=info["strategy_return"],
                exposure=info["e_final"],
            )
            obs_next = builder.build(step_idx=t + 1)

    The rolling counters are seeded zero on reset; the first few steps
    therefore expose near-zero drawdown / turnover / hit-rate, which is
    the same warm-up behaviour the inner ExposureEnv's risk state has.
    """

    def __init__(
        self,
        tape: CompactObservationTape,
        cfg: Optional[CompactObservationConfig] = None,
    ) -> None:
        if cfg is None:
            cfg = CompactObservationConfig()
        self._tape = tape
        self._cfg = cfg
        self._has_regime = (
            cfg.include_regime_one_hot and tape.regime_one_hot is not None
        )
        self._state = _RollingState()

    @property
    def cfg(self) -> CompactObservationConfig:
        return self._cfg

    @property
    def tape(self) -> CompactObservationTape:
        return self._tape

    @property
    def has_regime(self) -> bool:
        return bool(self._has_regime)

    @property
    def obs_dim(self) -> int:
        """Total observation dimensionality."""
        dim = N_BASE_FIELDS
        if self._has_regime:
            dim += N_REGIME_CLUSTERS
        # e_star is always appended last (mirrors KellyPriorEnvWrapper).
        dim += 1
        return int(dim)

    def reset(self) -> None:
        """Reset rolling counters at the start of a new episode."""
        self._state = _RollingState(
            return_hist=deque(maxlen=max(
                self._cfg.drawdown_window,
                self._cfg.hit_rate_window,
            )),
            turnover_hist=deque(maxlen=self._cfg.turnover_window),
            hit_hist=deque(maxlen=self._cfg.hit_rate_window),
        )

    def update(
        self,
        step_idx: int,
        strategy_return: float,
        exposure: float,
    ) -> None:
        """Update rolling counters after a step's outcome is known.

        Args:
            step_idx: Step index within the episode (0-based). Used only
                to range-check against the tape.
            strategy_return: Realised strategy return on this step.
            exposure: Final exposure applied on this step (post-clip).
        """
        T = len(self._tape)
        if step_idx < 0 or step_idx >= T:
            raise ValueError(
                f"[ERR] update(step_idx={step_idx}) out of range [0, {T})"
            )
        r = float(strategy_return)
        e = float(exposure)
        if not np.isfinite(r):
            raise ValueError(f"[ERR] strategy_return is non-finite: {r}")
        if not np.isfinite(e):
            raise ValueError(f"[ERR] exposure is non-finite: {e}")
        # Equity + drawdown.
        st = self._state
        st.equity *= 1.0 + r
        if st.equity > st.hwm:
            st.hwm = st.equity
        # Rolling histories.
        st.return_hist.append(r)
        st.turnover_hist.append(abs(e - st.prev_exposure))
        st.hit_hist.append(1 if r > 0.0 else 0)
        st.prev_exposure = e

    def _drawdown(self) -> float:
        st = self._state
        if st.hwm <= 0.0:
            return 0.0
        return float(max(0.0, 1.0 - st.equity / st.hwm))

    def _turnover(self) -> float:
        if len(self._state.turnover_hist) == 0:
            return 0.0
        return float(np.mean(self._state.turnover_hist))

    def _hit_rate(self) -> float:
        if len(self._state.hit_hist) == 0:
            return 0.0
        return float(np.mean(self._state.hit_hist))

    def _at(self, idx: int) -> int:
        """Clamp ``idx`` into the tape range [0, T-1]."""
        T = len(self._tape)
        return int(max(0, min(T - 1, idx)))

    def build(self, step_idx: int) -> np.ndarray:
        """Assemble the compact observation for ``step_idx``."""
        i = self._at(step_idx)
        t = self._tape
        p_hat = float(t.p_hat[i])
        mu_hat = float(t.mu_hat[i])
        sigma_hat = float(t.sigma_hat[i])
        drawdown = self._drawdown()
        turnover = self._turnover()
        hit_rate = self._hit_rate()
        vix_norm = float(t.vix_per_day[i]) / float(self._cfg.vix_scale)
        ust10y_norm = (
            float(t.ust10y_per_day[i]) / float(self._cfg.ust10y_scale)
        )
        base = np.asarray(
            [
                p_hat,
                mu_hat,
                sigma_hat,
                drawdown,
                turnover,
                hit_rate,
                vix_norm,
                ust10y_norm,
            ],
            dtype=np.float32,
        )
        e_star = float(t.e_star[i])
        if self._has_regime:
            regime = t.regime_one_hot[i].astype(np.float32, copy=False)
            obs = np.concatenate(
                [base, regime, np.asarray([e_star], dtype=np.float32)]
            )
        else:
            obs = np.concatenate(
                [base, np.asarray([e_star], dtype=np.float32)]
            )
        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(
                f"[ERR] obs dim mismatch: got {obs.shape[0]} expected "
                f"{self.obs_dim}"
            )
        if not np.isfinite(obs).all():
            raise RuntimeError(
                f"[ERR] compact obs contains NaN/inf at step {step_idx}"
            )
        return obs


def build_regime_one_hot(
    tape_days: np.ndarray,
    day_to_cluster: dict,
    n_clusters: int = N_REGIME_CLUSTERS,
) -> np.ndarray:
    """Build a ``(T, n_clusters)`` one-hot from a day_idx -> cluster map.

    Days not present in the lookup are encoded as an all-zero row (NOT
    an extra column); this is the conservative behaviour. The driver is
    expected to report any per-tape miss rate as a diagnostic so silent
    coverage gaps cannot mask issues.

    Args:
        tape_days: ``(T,)`` array of global trading-day indices.
        day_to_cluster: dict ``int -> int`` cluster id in
            ``[0, n_clusters)``.
        n_clusters: Number of clusters (default 8).

    Returns:
        ``(T, n_clusters)`` float32 array.
    """
    if n_clusters < 1:
        raise ValueError(
            f"[ERR] n_clusters must be >= 1; got {n_clusters}"
        )
    T = int(tape_days.shape[0])
    out = np.zeros((T, int(n_clusters)), dtype=np.float32)
    for i, d in enumerate(tape_days.tolist()):
        c = day_to_cluster.get(int(d))
        if c is None:
            continue
        c_int = int(c)
        if c_int < 0 or c_int >= n_clusters:
            raise ValueError(
                f"[ERR] cluster id {c_int} out of range [0, {n_clusters}) "
                f"for day_idx={d}"
            )
        out[i, c_int] = 1.0
    return out


__all__ = [
    "N_BASE_FIELDS",
    "N_REGIME_CLUSTERS",
    "DRAWDOWN_WINDOW_DAYS",
    "TURNOVER_WINDOW_DAYS",
    "HIT_RATE_WINDOW_DAYS",
    "CompactObservationConfig",
    "CompactObservationTape",
    "CompactObservationBuilder",
    "build_regime_one_hot",
]
