"""Non-learning baseline strategies for the InVAR-RL comparison table.

Every baseline produces a per-day log-return series on the same test
segment InVAR-RL is evaluated on, so the comparison metric (annualised
Sharpe + final equity) is computed identically:

    ann_ret  = mean(daily_log_ret) * 252
    ann_vol  = std(daily_log_ret) * sqrt(252)
    sharpe   = ann_ret / ann_vol
    final_eq = exp(sum(daily_log_ret))

All baselines pull data exclusively from
:class:`invar_rl.data.lattice_bridge.LatticePanelBatch`, so the
universe, dates, and active-stock mask are identical to those used by
the canonical InVAR-RL pipeline.

Baselines implemented:
- buy_and_hold: long-only, fix weights at start of test, no rebalance.
- equal_weight_long: long-only, equal weight across all active stocks
  each day (daily rebalance). Standard "market" benchmark on the
  panel universe.
- momentum_long_short: dollar-neutral L/S, rank by past
  (lookback - skip) log return, long top decile, short bottom decile,
  monthly rebalance. Jegadeesh-Titman 1993 with the standard 12-2
  configuration.
- reversal_long_short: dollar-neutral L/S, opposite of momentum on
  short horizon. Lo & MacKinlay 1990. lookback=21 days.
- volatility_targeted_market: equal-weight long-only with daily
  exposure scaled to hit an annualised vol target. Crude FF-like
  market-factor scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from invar_rl.data.lattice_bridge import LatticePanelBatch


@dataclass
class BaselineResult:
    """Per-day log-return + summary metrics for one baseline."""

    name: str
    daily_log_returns: np.ndarray
    mean_return: float
    volatility: float
    sharpe_annualised: float
    final_equity: float
    n_days: int
    gross_exposure_mean: float
    net_exposure_mean: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "mean_return": float(self.mean_return),
            "volatility": float(self.volatility),
            "sharpe_annualised": float(self.sharpe_annualised),
            "final_equity": float(self.final_equity),
            "n_days": int(self.n_days),
            "gross_exposure_mean": float(self.gross_exposure_mean),
            "net_exposure_mean": float(self.net_exposure_mean),
        }


def _summarise(
    name: str,
    daily: np.ndarray,
    gross: np.ndarray,
    net: np.ndarray,
) -> BaselineResult:
    daily = np.asarray(daily, dtype=np.float64)
    daily = daily[np.isfinite(daily)]
    if daily.size == 0:
        return BaselineResult(
            name=name,
            daily_log_returns=np.zeros(0),
            mean_return=0.0,
            volatility=0.0,
            sharpe_annualised=0.0,
            final_equity=1.0,
            n_days=0,
            gross_exposure_mean=0.0,
            net_exposure_mean=0.0,
        )
    mean = float(daily.mean())
    vol = float(daily.std(ddof=1)) if daily.size > 1 else 0.0
    ann_ret = mean * 252.0
    ann_vol = vol * np.sqrt(252.0)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    equity = float(np.exp(daily.sum()))
    return BaselineResult(
        name=name,
        daily_log_returns=daily,
        mean_return=mean,
        volatility=vol,
        sharpe_annualised=sharpe,
        final_equity=equity,
        n_days=int(daily.size),
        gross_exposure_mean=float(np.mean(gross)),
        net_exposure_mean=float(np.mean(net)),
    )


def buy_and_hold(
    bridge: LatticePanelBatch, day_indices: Sequence[int]
) -> BaselineResult:
    """Equal-weight on day 0 of test, hold without rebalancing.

    Each day's portfolio log return is the equal-weighted mean of the
    active stocks selected on day 0; if a stock becomes inactive its
    return is set to zero (no recovery).
    """
    days = list(day_indices)
    if not days:
        return _summarise("buy_and_hold", np.zeros(0), [0.0], [0.0])
    start = days[0]
    held = np.nonzero(bridge.tradable[start])[0]
    if held.size == 0:
        return _summarise("buy_and_hold", np.zeros(0), [0.0], [0.0])
    w = np.ones(held.size, dtype=np.float64) / held.size
    daily = []
    gross_hist = []
    net_hist = []
    for d in days:
        if d + 1 >= bridge.log_returns_1d.shape[0]:
            break
        r = bridge.log_returns_1d[d + 1, held]
        r = np.where(np.isfinite(r), r, 0.0)
        daily.append(float((w * r).sum()))
        gross_hist.append(float(np.sum(np.abs(w))))
        net_hist.append(float(np.sum(w)))
    return _summarise(
        "buy_and_hold",
        np.asarray(daily),
        np.asarray(gross_hist),
        np.asarray(net_hist),
    )


def equal_weight_long(
    bridge: LatticePanelBatch, day_indices: Sequence[int]
) -> BaselineResult:
    """Long-only equal weight, daily rebalance over the active set."""
    daily = []
    gross_hist = []
    net_hist = []
    for d in day_indices:
        if d + 1 >= bridge.log_returns_1d.shape[0]:
            break
        active = np.nonzero(bridge.tradable[d])[0]
        if active.size == 0:
            continue
        r = bridge.log_returns_1d[d + 1, active]
        r = np.where(np.isfinite(r), r, 0.0)
        daily.append(float(r.mean()))
        gross_hist.append(1.0)
        net_hist.append(1.0)
    return _summarise(
        "equal_weight_long",
        np.asarray(daily),
        np.asarray(gross_hist),
        np.asarray(net_hist),
    )


def momentum_long_short(
    bridge: LatticePanelBatch,
    day_indices: Sequence[int],
    lookback: int = 252,
    skip: int = 21,
    decile: float = 0.1,
    rebalance_days: int = 21,
) -> BaselineResult:
    """Jegadeesh-Titman 12-2 momentum, decile spread, monthly rebalance.

    On each rebalance day, rank stocks by their cumulative log return
    over [d - lookback, d - skip]; long the top decile equally and
    short the bottom decile equally (gross = 1, net = 0). Hold the
    weights until the next rebalance.
    """
    daily = []
    gross_hist = []
    net_hist = []
    w_current = None
    held_current = None
    days_since_reb = rebalance_days
    for d in day_indices:
        if d + 1 >= bridge.log_returns_1d.shape[0]:
            break
        if days_since_reb >= rebalance_days:
            lo = max(0, d - lookback)
            hi = max(lo + 1, d - skip)
            past = bridge.log_returns_1d[lo:hi, :]
            past = np.where(np.isfinite(past), past, 0.0)
            cum = past.sum(axis=0)
            active = bridge.tradable[d]
            cum = np.where(active, cum, np.nan)
            valid = np.isfinite(cum)
            if valid.sum() < 20:
                w_current = None
            else:
                k = max(1, int(decile * valid.sum()))
                idx_sorted = np.argsort(np.where(valid, cum, -np.inf))
                short = idx_sorted[:k]
                long = idx_sorted[-k:]
                w_full = np.zeros(bridge.log_returns_1d.shape[1])
                w_full[long] = 1.0 / (2.0 * k)
                w_full[short] = -1.0 / (2.0 * k)
                held_current = np.nonzero(np.abs(w_full) > 0)[0]
                w_current = w_full[held_current]
            days_since_reb = 0
        days_since_reb += 1
        if w_current is None or held_current is None:
            continue
        r = bridge.log_returns_1d[d + 1, held_current]
        r = np.where(np.isfinite(r), r, 0.0)
        daily.append(float((w_current * r).sum()))
        gross_hist.append(float(np.sum(np.abs(w_current))))
        net_hist.append(float(np.sum(w_current)))
    return _summarise(
        f"momentum_long_short_lb{lookback}_sk{skip}",
        np.asarray(daily),
        np.asarray(gross_hist) if gross_hist else np.asarray([0.0]),
        np.asarray(net_hist) if net_hist else np.asarray([0.0]),
    )


def reversal_long_short(
    bridge: LatticePanelBatch,
    day_indices: Sequence[int],
    lookback: int = 21,
    decile: float = 0.1,
    rebalance_days: int = 5,
) -> BaselineResult:
    """Short-horizon reversal, opposite of momentum (Lo & MacKinlay 1990).

    Long bottom-decile past-lookback performers, short top decile.
    Weekly rebalance by default.
    """
    daily = []
    gross_hist = []
    net_hist = []
    w_current = None
    held_current = None
    days_since_reb = rebalance_days
    for d in day_indices:
        if d + 1 >= bridge.log_returns_1d.shape[0]:
            break
        if days_since_reb >= rebalance_days:
            lo = max(0, d - lookback)
            past = bridge.log_returns_1d[lo:d, :]
            past = np.where(np.isfinite(past), past, 0.0)
            cum = past.sum(axis=0)
            active = bridge.tradable[d]
            cum = np.where(active, cum, np.nan)
            valid = np.isfinite(cum)
            if valid.sum() < 20:
                w_current = None
            else:
                k = max(1, int(decile * valid.sum()))
                idx_sorted = np.argsort(np.where(valid, cum, -np.inf))
                short = idx_sorted[-k:]  # past winners
                long = idx_sorted[:k]    # past losers
                w_full = np.zeros(bridge.log_returns_1d.shape[1])
                w_full[long] = 1.0 / (2.0 * k)
                w_full[short] = -1.0 / (2.0 * k)
                held_current = np.nonzero(np.abs(w_full) > 0)[0]
                w_current = w_full[held_current]
            days_since_reb = 0
        days_since_reb += 1
        if w_current is None or held_current is None:
            continue
        r = bridge.log_returns_1d[d + 1, held_current]
        r = np.where(np.isfinite(r), r, 0.0)
        daily.append(float((w_current * r).sum()))
        gross_hist.append(float(np.sum(np.abs(w_current))))
        net_hist.append(float(np.sum(w_current)))
    return _summarise(
        f"reversal_long_short_lb{lookback}",
        np.asarray(daily),
        np.asarray(gross_hist) if gross_hist else np.asarray([0.0]),
        np.asarray(net_hist) if net_hist else np.asarray([0.0]),
    )


def volatility_targeted_market(
    bridge: LatticePanelBatch,
    day_indices: Sequence[int],
    target_ann_vol: float = 0.10,
    vol_lookback: int = 60,
) -> BaselineResult:
    """Equal-weight long-only with daily exposure scaled to a vol target.

    Used as a "smart market" baseline: hold the diversified equal-weight
    market portfolio, but scale gross exposure each day to match the
    annualised volatility target estimated from the trailing
    ``vol_lookback`` days of the market's own return.
    """
    daily = []
    gross_hist = []
    net_hist = []
    realised_market: list[float] = []
    for d in day_indices:
        if d + 1 >= bridge.log_returns_1d.shape[0]:
            continue
        active = np.nonzero(bridge.tradable[d])[0]
        if active.size == 0:
            continue
        if len(realised_market) >= vol_lookback:
            window = np.asarray(realised_market[-vol_lookback:])
            current_ann_vol = float(window.std(ddof=1)) * np.sqrt(252.0)
            exposure = (
                target_ann_vol / current_ann_vol if current_ann_vol > 0
                else 1.0
            )
        else:
            exposure = 1.0
        exposure = float(np.clip(exposure, 0.0, 1.5))
        r_next = bridge.log_returns_1d[d + 1, active]
        r_next = np.where(np.isfinite(r_next), r_next, 0.0)
        market_ret = float(r_next.mean())
        realised_market.append(market_ret)
        daily.append(exposure * market_ret)
        gross_hist.append(exposure)
        net_hist.append(exposure)
    return _summarise(
        f"vol_targeted_market_tgt{int(target_ann_vol*100)}",
        np.asarray(daily),
        np.asarray(gross_hist) if gross_hist else np.asarray([0.0]),
        np.asarray(net_hist) if net_hist else np.asarray([0.0]),
    )


__all__ = [
    "BaselineResult",
    "buy_and_hold",
    "equal_weight_long",
    "momentum_long_short",
    "reversal_long_short",
    "volatility_targeted_market",
]
