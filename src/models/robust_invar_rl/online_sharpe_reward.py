"""Robust-InVAR-RL Phase 3: EWMA-based online Sharpe reward.

The canonical Layer 3 reward sums per-step PnL with optional risk
penalties (see :mod:`invar_rl.layer3_control.reward`). Per the source
design doc (2026-05-26), this objective is not well aligned with the
actual evaluation metric (annualised Sharpe of the test daily-return
tape) and contributes to SAC over-control.

The online Sharpe reward replaces the per-step PnL with the change
in an EWMA-based Sharpe estimator. Formally::

    alpha          = 1 - 0.5 ** (1 / half_life_days)
    mu_old, v_old  = state before step t
    mu_new         = alpha * r_t + (1 - alpha) * mu_old
    v_new          = alpha * (r_t - mu_new)^2 + (1 - alpha) * v_old
    sigma_new      = max(sqrt(v_new), eps)
    reward_t       = (mu_new - mu_old) / sigma_new * sqrt(252)

Warm-up: during the first ``warmup_steps`` evaluations the variance
estimator is not yet meaningful (sigma is tiny + dominated by the very
first squared deviation). Returning the raw daily Sharpe-equivalent
``r_t * sqrt(252)`` for these steps avoids a divide-by-tiny-sigma blow
up while still giving the SAC actor a smooth reward signal.

The reward is bounded by a runtime ``clip`` (default +/- 8 in Sharpe
units) so a pathological single day cannot saturate the SAC value
network.

State is held inside the class; the caller must invoke :meth:`reset`
once per episode and :meth:`step` exactly once per environment step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


_LOG_PREFIX = "[Phase3-OnlineSharpe]"

# Canonical trading-days-per-year convention used in the rest of the
# InVAR-RL stack (see invar_rl.layer3_control.reward).
TRADING_DAYS_PER_YEAR: float = 252.0


@dataclass
class OnlineSharpeRewardConfig:
    """Hyperparameters for :class:`OnlineSharpeReward`."""

    half_life_days: int = 21
    eps: float = 1.0e-6
    warmup_steps: int = 5
    clip: float = 8.0

    def __post_init__(self) -> None:
        if self.half_life_days < 1:
            raise ValueError(
                "[ERR] half_life_days must be >= 1; "
                f"got {self.half_life_days}"
            )
        if self.eps <= 0.0:
            raise ValueError(
                f"[ERR] eps must be > 0; got {self.eps}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"[ERR] warmup_steps must be >= 0; got {self.warmup_steps}"
            )
        if self.clip <= 0.0:
            raise ValueError(
                f"[ERR] clip must be > 0; got {self.clip}"
            )


class OnlineSharpeReward:
    """Stateful EWMA Sharpe-increment reward.

    Equivalent to the discrete-time derivative of an EWMA estimate of
    annualised Sharpe. Bounded, finite-state, and deterministic given
    the same per-step inputs.
    """

    def __init__(
        self,
        cfg: Optional[OnlineSharpeRewardConfig] = None,
    ) -> None:
        if cfg is None:
            cfg = OnlineSharpeRewardConfig()
        self._cfg = cfg
        self._alpha = float(
            1.0 - 0.5 ** (1.0 / float(cfg.half_life_days))
        )
        self._sqrt_year = float(np.sqrt(TRADING_DAYS_PER_YEAR))
        self._mu: float = 0.0
        self._var: float = 0.0
        self._n_steps: int = 0
        self._init: bool = False

    @property
    def cfg(self) -> OnlineSharpeRewardConfig:
        return self._cfg

    @property
    def alpha(self) -> float:
        return float(self._alpha)

    @property
    def n_steps(self) -> int:
        return int(self._n_steps)

    @property
    def mu(self) -> float:
        return float(self._mu)

    @property
    def var(self) -> float:
        return float(self._var)

    def reset(self) -> None:
        """Reset the EWMA state at the start of an episode."""
        self._mu = 0.0
        self._var = 0.0
        self._n_steps = 0
        self._init = False

    def step(self, daily_return: float) -> float:
        """Consume one realised daily return and emit a reward.

        Args:
            daily_return: Realised strategy return for the step.

        Returns:
            A finite scalar in ``[-clip, +clip]``.
        """
        r = float(daily_return)
        if not np.isfinite(r):
            raise ValueError(
                f"[ERR] daily_return is non-finite: {r}"
            )
        cfg = self._cfg
        self._n_steps += 1
        # Warm-up: emit raw Sharpe-equivalent scaled return.
        if self._n_steps <= cfg.warmup_steps:
            # Still update internal state so warm-up is not wasted.
            if not self._init:
                self._mu = r
                self._var = 0.0
                self._init = True
            else:
                mu_new = self._alpha * r + (1.0 - self._alpha) * self._mu
                dev2 = (r - mu_new) ** 2
                self._var = (
                    self._alpha * dev2 + (1.0 - self._alpha) * self._var
                )
                self._mu = mu_new
            raw = float(r * self._sqrt_year)
            return float(np.clip(raw, -cfg.clip, cfg.clip))

        mu_old = float(self._mu)
        mu_new = self._alpha * r + (1.0 - self._alpha) * mu_old
        dev2 = (r - mu_new) ** 2
        var_new = self._alpha * dev2 + (1.0 - self._alpha) * self._var
        sigma_new = float(np.sqrt(max(var_new, 0.0)))
        sigma_eff = max(sigma_new, cfg.eps)
        reward = (mu_new - mu_old) / sigma_eff * self._sqrt_year
        self._mu = float(mu_new)
        self._var = float(var_new)
        if not np.isfinite(reward):
            raise RuntimeError(
                "[ERR] online Sharpe reward produced non-finite value; "
                f"mu_old={mu_old} mu_new={mu_new} sigma_eff={sigma_eff} "
                f"r={r}"
            )
        return float(np.clip(float(reward), -cfg.clip, cfg.clip))


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "OnlineSharpeRewardConfig",
    "OnlineSharpeReward",
]
