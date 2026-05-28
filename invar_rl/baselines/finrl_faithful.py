"""FinRL StockTradingEnv aligned to AI4Finance-Foundation/FinRL master.

Reproduces the canonical FinRL setup from Liu et al. (NeurIPS 2020 DRL
Workshop) and Yang et al. (ICAIF 2020), aligned to the upstream GitHub
repo as of 2026-05-21:

- 30 canonical DJIA constituents from ``finrl/config_tickers.py``
  (``DOW_30_TICKER``).
- 8 canonical indicators from ``finrl/config.py`` (``INDICATORS``):
  macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma,
  close_60_sma.
- Per-day VIX column added to the observation (single scalar
  broadcast across all tickers per day, matching upstream
  ``FeatureEngineer(use_vix=True)``).
- State layout: ``[balance, shares (N), close (N), 8 indicators (8N),
  VIX (1)]`` = 1 + N + 8N + 1 = 2 + 9N. For N=30: 272-d.
- Action space: Box(-1, 1, N), multiplied by hmax=100, int cast.
- Transaction cost: 0.1% per leg, dollar-value-multiplicative.
- Reward: change in total asset value, times reward_scaling=1e-4.
- Turbulence overlay: when day's Mahalanobis turbulence > threshold,
  liquidate ALL positions to cash this step (paper-faithful;
  documented divergence from the repo, which rate-limits via
  ``-hmax`` per leg per day).
- Sell/buy loop ordered by argsort of absolute action magnitude
  (largest |action| first, sells before buys), matching upstream
  ``env_stocktrading.py``.

PPO/A2C/DDPG hyperparameters aligned to ``finrl/config.py``:
  PPO: n_steps=2048, ent_coef=0.01, lr=2.5e-4, batch_size=128.
  A2C: n_steps=5,    ent_coef=0.01, lr=7e-4.
  DDPG: lr=1e-3, buffer_size=1_000_000, NormalActionNoise(sigma=0.1).

Universe: DJIA-30 canonical list (DOW_30_TICKER). Date range:
configurable; default 2009-01-01 to 2020-06-30 matches the ICAIF 2020
paper window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


# Canonical DOW_30_TICKER from AI4Finance-Foundation/FinRL master
# (finrl/config_tickers.py). Replaces our previous broken list (28 unique
# with WBA duplicated, AMGN/NVDA/CRM/SHW missing, DOW spurious).
DJIA_30_TICKERS = (
    "AXP", "AMGN", "AMZN", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS",
    "HD", "HON", "IBM", "JNJ", "KO", "JPM", "MCD", "MMM", "MRK",
    "MSFT", "NVDA", "NKE", "PG", "TRV", "UNH", "CRM", "VZ", "V",
    "WMT", "DIS", "SHW",
)


# Canonical INDICATORS list from finrl/config.py (8 items including
# Bollinger upper/lower bands).
FINRL_INDICATORS = (
    "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30",
    "close_30_sma", "close_60_sma",
)


def _technical_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute FinRL-standard technical indicators per ticker.

    Inputs ``prices`` with columns: ticker, date, open, high, low, close,
    volume. Returns the same DataFrame with one column per indicator in
    ``FINRL_INDICATORS`` (close, MACD, RSI-30, CCI-30, ADX-30,
    close_30_sma, close_60_sma; the FinRL default set).
    """
    from stockstats import wrap as ss_wrap
    out = []
    for ticker, group in prices.sort_values(["ticker", "date"]).groupby("ticker"):
        df = group.reset_index(drop=True).rename(
            columns={"date": "date"}
        )
        try:
            stock = ss_wrap(df.copy())
            for ind in FINRL_INDICATORS:
                df[ind] = stock[ind].values.astype(np.float64)
        except Exception:
            for ind in FINRL_INDICATORS:
                df[ind] = 0.0
        out.append(df)
    return pd.concat(out, ignore_index=True).sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)


def _fetch_vix(start: str, end: str) -> pd.Series:
    """Fetch CBOE VIX close via yfinance ``^VIX`` over [start, end].

    Returns a Series indexed by date (DatetimeIndex) with the VIX close.
    Matches upstream FinRL ``FeatureEngineer.add_vix``: one per-day
    scalar broadcast across all tickers. Caller is responsible for
    aligning to the trading-day index and forward-filling gaps.
    """
    import yfinance as yf
    try:
        df = yf.download(
            "^VIX", start=start, end=end, auto_adjust=False,
            progress=False, threads=False,
        )
    except Exception:
        return pd.Series(dtype=np.float64)
    if df.empty:
        return pd.Series(dtype=np.float64)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    if "close" not in df.columns:
        return pd.Series(dtype=np.float64)
    s = pd.Series(df["close"].astype(np.float64).values,
                  index=pd.to_datetime(df["date"]))
    s.name = "vix"
    return s


def _turbulence_index(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    """Mahalanobis-distance turbulence index per date, FinRL convention."""
    pivot = prices.pivot(
        index="date", columns="ticker", values="close"
    ).sort_index()
    returns = pivot.pct_change().fillna(0.0)
    dates = returns.index
    out = pd.Series(0.0, index=dates, dtype=np.float64)
    for i, d in enumerate(dates):
        if i < window:
            continue
        hist = returns.iloc[i - window: i]
        mu = hist.mean(axis=0).values
        try:
            cov = np.cov(hist.values.T)
            cov_inv = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))
            current = returns.iloc[i].values - mu
            out.iloc[i] = float(current @ cov_inv @ current)
        except Exception:
            out.iloc[i] = 0.0
    return out


@dataclass
class FinRLEnvConfig:
    initial_balance: float = 1_000_000.0
    hmax: int = 100
    transaction_cost_pct: float = 0.001
    reward_scaling: float = 1e-4
    turbulence_threshold: float = 140.0
    # State dim is computed at env construction (1 + N + 8N + 1 for the
    # canonical 8-indicator + VIX layout). Stored for downstream
    # diagnostics; not used inside the env.
    state_dim: int = 0


class FinRLStockTradingEnv(gym.Env):
    """Canonical FinRL StockTradingEnv with long-only discrete shares.

    State layout (aligned to upstream env_stocktrading.py):
        - balance (1)
        - shares held per ticker (N)
        - close price per ticker (N)
        - 8 indicators per ticker (8N) [macd, boll_ub, boll_lb,
          rsi_30, cci_30, dx_30, close_30_sma, close_60_sma]
        - VIX (1) [single scalar per day, matches upstream
          FeatureEngineer(use_vix=True)]
        = 1 + N + N + 8N + 1 = 2 + 10N. For N=30 -> 302.

    Action: a continuous vector in [-1, 1]^N. Each entry is multiplied
    by hmax and integer-cast to the number of shares to buy (+) or sell
    (-). Buys are skipped if balance is insufficient; sells are skipped
    if shares are zero. Transaction cost is 0.1% of the dollar value
    traded per leg. Sells and buys are processed in argsort order of
    |action|, sells (negative actions, largest magnitude first) before
    buys (positive actions, largest magnitude first), matching
    upstream's ordering in env_stocktrading.py.

    Reward: change in total asset value (balance + shares * close)
    times ``reward_scaling`` (=1e-4 in the published recipe).

    Turbulence overlay (deliberate divergence from upstream): if the
    day's Mahalanobis turbulence index exceeds
    ``cfg.turbulence_threshold``, all positions are liquidated to cash
    in a single step. Upstream env_stocktrading.py uses
    ``actions = [-hmax] * N`` and lets the normal sell-then-buy
    machinery run, which rate-limits liquidation to hmax shares per
    name per day. We follow the ICAIF 2020 paper wording ("liquidate
    all positions") rather than the repo's rate-limited
    implementation. The behaviour is intentional and noted here for
    reviewers diffing the two codebases.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        cfg: FinRLEnvConfig,
        indicators: Tuple[str, ...] = FINRL_INDICATORS,
        turbulence: Optional[pd.Series] = None,
        vix: Optional[pd.Series] = None,
    ) -> None:
        super().__init__()
        self._df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        self._tickers = list(tickers)
        self._N = len(self._tickers)
        self._indicators = tuple(indicators)
        self._cfg = cfg
        self._dates = sorted(self._df["date"].unique())
        self._T = len(self._dates)
        self._turbulence = turbulence
        # Pivot tables for fast per-day lookup. Forward-fill then
        # back-fill then 0 so the env never sees NaN.
        close_piv = self._df.pivot(
            index="date", columns="ticker", values="close"
        ).reindex(columns=self._tickers)
        close_piv = close_piv.ffill().bfill().fillna(1.0)
        self._close = close_piv.values.astype(np.float64)
        self._ind = {}
        for ind in self._indicators:
            piv = self._df.pivot(
                index="date", columns="ticker", values=ind
            ).reindex(columns=self._tickers).fillna(0.0)
            self._ind[ind] = piv.values.astype(np.float64)
        if self._turbulence is not None:
            # Align turbulence to env dates; fill NaN with 0 so no spurious
            # liquidation.
            self._turbulence_values = (
                self._turbulence.reindex(self._dates).fillna(0.0).values
            )
        else:
            self._turbulence_values = None
        # Single per-day VIX column (matches upstream's
        # FeatureEngineer.add_vix). Align to env dates and forward-fill
        # any missing values, then back-fill any leading gap, then 0.
        if vix is not None and not vix.empty:
            self._vix_values = (
                vix.reindex(self._dates).ffill().bfill().fillna(0.0).values
                .astype(np.float64)
            )
        else:
            self._vix_values = np.zeros(self._T, dtype=np.float64)
        obs_dim = 1 + self._N + self._N + len(self._indicators) * self._N + 1
        cfg.state_dim = int(obs_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._N,), dtype=np.float32,
        )
        self._t = 0
        self._balance = float(cfg.initial_balance)
        self._shares = np.zeros(self._N, dtype=np.int64)

    def _state(self) -> np.ndarray:
        prices = self._close[self._t]
        parts = [
            np.array([self._balance], dtype=np.float64),
            self._shares.astype(np.float64),
            prices.astype(np.float64),
        ]
        for ind in self._indicators:
            parts.append(self._ind[ind][self._t].astype(np.float64))
        parts.append(np.array([self._vix_values[self._t]], dtype=np.float64))
        return np.nan_to_num(
            np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0,
        ).astype(np.float32)

    def _total_asset(self) -> float:
        return float(
            self._balance + (self._shares * self._close[self._t]).sum()
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._balance = float(self._cfg.initial_balance)
        self._shares = np.zeros(self._N, dtype=np.int64)
        return self._state(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        # Turbulence overlay: liquidate everything to cash if today
        # exceeds threshold.
        if self._turbulence_values is not None:
            today_turb = float(self._turbulence_values[self._t])
            if today_turb > self._cfg.turbulence_threshold:
                prices = self._close[self._t]
                proceeds = (self._shares * prices).sum()
                cost = proceeds * self._cfg.transaction_cost_pct
                self._balance += float(proceeds - cost)
                self._shares = np.zeros(self._N, dtype=np.int64)
                # No new action this step.
                pre_asset = self._total_asset()
                self._t += 1
                if self._t >= self._T - 1:
                    return (
                        self._state(),
                        0.0,
                        True, False,
                        {
                            "total_asset": self._total_asset(),
                            "turbulence": today_turb,
                        },
                    )
                post_asset = self._total_asset()
                reward = (post_asset - pre_asset) * self._cfg.reward_scaling
                return (
                    self._state(),
                    float(reward),
                    False, False,
                    {
                        "total_asset": post_asset,
                        "turbulence": today_turb,
                        "liquidated": True,
                    },
                )
        target_shares = (action * self._cfg.hmax).astype(np.int64)
        prices = self._close[self._t]
        # Argsort by |action|: process the largest-magnitude trades
        # first, sells before buys. Matches upstream
        # env_stocktrading.py, where a balance-binding day buys the
        # most confident positions first rather than left-to-right by
        # ticker index.
        order = np.argsort(-np.abs(action))
        for i in order:
            delta = int(target_shares[i])
            if delta < 0 and self._shares[i] > 0:
                sell_n = min(int(-delta), int(self._shares[i]))
                proceeds = sell_n * prices[i]
                cost = proceeds * self._cfg.transaction_cost_pct
                self._balance += float(proceeds - cost)
                self._shares[i] -= sell_n
        for i in order:
            delta = int(target_shares[i])
            if delta > 0:
                pi = float(prices[i])
                if not np.isfinite(pi) or pi <= 0:
                    continue
                max_affordable = int(self._balance // (
                    pi * (1.0 + self._cfg.transaction_cost_pct)
                ))
                buy_n = min(int(delta), max(0, max_affordable))
                if buy_n > 0:
                    cost = buy_n * prices[i]
                    fee = cost * self._cfg.transaction_cost_pct
                    self._balance -= float(cost + fee)
                    self._shares[i] += buy_n
        pre_asset = self._total_asset()
        self._t += 1
        if self._t >= self._T - 1:
            return (
                self._state(),
                0.0,
                True, False,
                {"total_asset": self._total_asset()},
            )
        post_asset = self._total_asset()
        reward = (post_asset - pre_asset) * self._cfg.reward_scaling
        return (
            self._state(),
            float(reward),
            False, False,
            {"total_asset": post_asset},
        )


def _summarise_episode(daily_assets: List[float], initial: float) -> Dict:
    arr = np.asarray(daily_assets, dtype=np.float64)
    if arr.size < 2:
        return {
            "ann_return": 0.0, "ann_vol": 0.0,
            "sharpe_annualised": 0.0,
            "final_equity": 1.0, "n_steps": int(arr.size),
        }
    log_rets = np.diff(np.log(arr))
    mean = float(log_rets.mean())
    vol = float(log_rets.std(ddof=1)) if log_rets.size > 1 else 0.0
    ann_ret = mean * 252.0
    ann_vol = vol * np.sqrt(252.0)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    return {
        "mean_return": mean,
        "volatility": vol,
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe_annualised": sharpe,
        "final_equity": float(arr[-1] / initial),
        "n_steps": int(arr.size),
    }


def evaluate_finrl_env(env: FinRLStockTradingEnv, agent) -> Dict:
    obs, _ = env.reset(seed=0)
    daily_assets = [float(env._total_asset())]
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
        if "total_asset" in info:
            daily_assets.append(float(info["total_asset"]))
        if term or trunc:
            break
    perf = _summarise_episode(daily_assets, env._cfg.initial_balance)
    arr = np.asarray(daily_assets, dtype=np.float64)
    perf["daily_log_returns"] = (
        np.diff(np.log(np.clip(arr, 1e-6, None))).tolist() if arr.size >= 2 else []
    )
    return perf


def train_finrl_ppo(
    train_env: FinRLStockTradingEnv,
    seed: int,
    total_timesteps: int = 50_000,
    learning_rate: float = 2.5e-4,
    n_steps: int = 2048,
    device: str = "cpu",
):
    """Train PPO on the FinRL env. Hyperparameters match upstream
    ``finrl/config.py PPO_PARAMS`` (n_steps=2048, ent_coef=0.01,
    lr=2.5e-4, batch_size=128)."""
    from stable_baselines3 import PPO
    agent = PPO(
        "MlpPolicy", train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=128,
        ent_coef=0.01,
        seed=seed,
        verbose=0,
        device=device,
    )
    agent.learn(total_timesteps=total_timesteps)
    return agent


def train_finrl_a2c(
    train_env: FinRLStockTradingEnv,
    seed: int,
    total_timesteps: int = 50_000,
    device: str = "cpu",
):
    """Train A2C on the FinRL env. Hyperparameters match upstream
    ``finrl/config.py A2C_PARAMS`` (n_steps=5, ent_coef=0.01,
    lr=7e-4)."""
    from stable_baselines3 import A2C
    agent = A2C(
        "MlpPolicy", train_env,
        learning_rate=7e-4, n_steps=5,
        ent_coef=0.01,
        seed=seed, verbose=0, device=device,
    )
    agent.learn(total_timesteps=total_timesteps)
    return agent


def train_finrl_ddpg(
    train_env: FinRLStockTradingEnv,
    seed: int,
    total_timesteps: int = 50_000,
    device: str = "cpu",
):
    """Train DDPG on the FinRL env. Hyperparameters match upstream
    ``finrl/agents/stablebaselines3/models.py`` DDPG defaults:
    lr=1e-3, buffer_size=1_000_000, NormalActionNoise(sigma=0.1)
    on the action dimension (upstream wraps DDPG with the noise
    object in ``DRLAgent.get_model``)."""
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions, dtype=np.float64),
        sigma=0.1 * np.ones(n_actions, dtype=np.float64),
    )
    agent = DDPG(
        "MlpPolicy", train_env,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        action_noise=action_noise,
        seed=seed, verbose=0, device=device,
    )
    agent.learn(total_timesteps=total_timesteps)
    return agent


__all__ = [
    "DJIA_30_TICKERS",
    "FINRL_INDICATORS",
    "FinRLEnvConfig",
    "FinRLStockTradingEnv",
    "_technical_indicators",
    "_turbulence_index",
    "_fetch_vix",
    "_summarise_episode",
    "evaluate_finrl_env",
    "train_finrl_ppo",
    "train_finrl_a2c",
    "train_finrl_ddpg",
]
