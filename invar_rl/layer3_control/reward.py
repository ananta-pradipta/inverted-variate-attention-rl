"""Differential Sharpe ratio and penalty terms for the Layer 3 reward.

The base signal is the Moody and Saffell incremental Sharpe update, with an
alternative based on the conditional value at risk of the return
distribution. A drawdown penalty, a turnover penalty on the change in
exposure, and a transaction-cost charge proportional to traded notional are
subtracted. All weights come from configuration.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from invar_rl.common.config import Layer3Config


class DifferentialSharpe:
    """Moody and Saffell incremental differential Sharpe ratio.

    Maintains exponential moving averages A of returns and B of squared
    returns with decay eta. With dA = R - A_prev and dB = R^2 - B_prev, the
    increment is

        (B_prev * dA - 0.5 * A_prev * dB) / (B_prev - A_prev^2) ** 1.5

    The first observation seeds the averages and returns a zero increment.
    """

    def __init__(
        self,
        eta: float,
        variance_floor: float = 1e-8,
        clip: float | None = None,
    ) -> None:
        """Initialise the estimator.

        Args:
            eta: EMA decay in (0, 1).
            variance_floor: Lower bound on the variance proxy
                ``B - A**2`` before it enters the denominator. This stops
                near-constant return streams from producing exploding,
                penalty-incomparable increments.
            clip: Optional symmetric bound on the returned increment. None
                leaves the raw differential Sharpe unbounded (used by the
                standalone math check); the environment sets a finite bound
                so the reward is comparable across regimes.
        """
        if not 0.0 < eta < 1.0:
            raise ValueError("eta must be in (0, 1)")
        if variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive")
        if clip is not None and clip <= 0.0:
            raise ValueError("clip must be positive when set")
        self._eta = float(eta)
        self._var_floor = float(variance_floor)
        self._clip = None if clip is None else float(clip)
        self._a = 0.0
        self._b = 0.0
        self._init = False

    def reset(self) -> None:
        self._a = 0.0
        self._b = 0.0
        self._init = False

    def update(self, r: float) -> float:
        """Consume one return and return the differential Sharpe increment."""
        if not self._init:
            self._a = r
            self._b = r * r
            self._init = True
            return 0.0
        d_a = r - self._a
        d_b = r * r - self._b
        variance = max(self._b - self._a ** 2, self._var_floor)
        increment = (
            self._b * d_a - 0.5 * self._a * d_b
        ) / (variance ** 1.5)
        if self._clip is not None:
            increment = float(np.clip(increment, -self._clip, self._clip))
        self._a += self._eta * d_a
        self._b += self._eta * d_b
        return float(increment)


def conditional_value_at_risk(
    returns: np.ndarray, level: float
) -> float:
    """Left-tail CVaR at ``level`` (a positive loss magnitude).

    Args:
        returns: Realised returns.
        level: Tail probability in (0, 0.5).

    Returns:
        The mean of the worst ``level`` fraction of returns, sign-flipped so
        a deeper left tail is a larger positive number. Zero if too few
        observations.
    """
    if returns.size < 2:
        return 0.0
    q = np.quantile(returns, level)
    tail = returns[returns <= q]
    if tail.size == 0:
        return 0.0
    return float(-tail.mean())


class RewardFunction:
    """Composite reward: base signal minus the configured penalties."""

    def __init__(self, cfg: Layer3Config) -> None:
        self._cfg = cfg
        self._ds = DifferentialSharpe(
            cfg.ds_decay,
            variance_floor=cfg.ds_variance_floor,
            clip=cfg.ds_clip,
        )
        self._returns: Deque[float] = deque(maxlen=252)

    def reset(self) -> None:
        self._ds.reset()
        self._returns.clear()

    def __call__(
        self,
        strategy_return: float,
        drawdown: float,
        delta_exposure: float,
        traded_notional: float,
    ) -> float:
        """Compute the step reward.

        Args:
            strategy_return: Realised return of the exposure-scaled book.
            drawdown: Current drawdown from the running high-water mark, a
                non-negative fraction.
            delta_exposure: Absolute day-to-day change in exposure.
            traded_notional: Notional traded this step.

        Returns:
            The scalar reward.
        """
        self._returns.append(strategy_return)
        if self._cfg.reward_kind == "differential_sharpe":
            base = self._ds.update(strategy_return)
        else:  # cvar
            cvar = conditional_value_at_risk(
                np.asarray(self._returns), self._cfg.cvar_level
            )
            base = strategy_return - cvar

        penalty = (
            self._cfg.drawdown_penalty * drawdown
            + self._cfg.turnover_penalty * abs(delta_exposure)
            + (self._cfg.transaction_cost_bps * 1e-4)
            * abs(traded_notional)
        )
        return float(base - penalty)
