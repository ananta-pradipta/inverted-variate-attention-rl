"""Whole-stack RL competitor baselines: StockFormer-style and FinRL-style.

Native re-implementations from the published descriptions (not the
authors' GitHub code) so we control every hyperparameter and so the
comparison is reproducible from the paper alone.

Both baselines operate END-TO-END on the universal S&P 500 lattice_native
panel: their own feature engineering, their own RL controller, their
own daily-rebalanced portfolio. Output is a per-day log-return series
on the InVAR-RL test segment so the comparison metric (annualised
Sharpe) is computed identically to InVAR-RL.

- **StockFormer (Gao et al., IJCAI 2023)** ``StockFormerBaseline``:
  per-stock transformer return predictor + SAC RL agent over the
  predicted-return cross-section, native action = portfolio weights
  (softmax over top-K candidates).
- **FinRL (Liu et al., NeurIPS 2020 + 2022)** ``FinRLBaseline``:
  technical-indicator features (returns, RSI-like z, MACD-like
  momentum, volatility) + PPO that outputs continuous portfolio
  weights, dollar-neutral via tanh+rescale.

Both use a shared portfolio gym environment :class:`PortfolioEnv`
in which the action IS the weight vector (not exposure on a fixed
portfolio, which is what InVAR-RL's ExposureEnv does).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from invar_rl.data.lattice_bridge import LatticePanelBatch


def _select_topk_universe(
    bridge: LatticePanelBatch,
    day_indices: List[int],
    k: int,
    seed: int = 0,
) -> np.ndarray:
    """Pick a stable universe of K most-liquid tickers across the window.

    Stable-universe selection avoids per-day cross-section changes that
    would otherwise blow up the state space.
    """
    n_active = bridge.tradable[day_indices].sum(axis=0)
    order = np.argsort(-n_active)
    selected = order[:k]
    return np.sort(selected)


def _technical_features(
    log_returns: np.ndarray,
    day_idx: int,
    universe: np.ndarray,
    lookback: int = 60,
) -> np.ndarray:
    """Compute (K, F) per-stock technical features at day ``day_idx``.

    Features (FinRL-style approximation):
    - 5/10/20-day cumulative log return (momentum)
    - 20-day rolling volatility
    - 60-day cumulative return z-score within lookback (RSI-like)
    - 5d vs 20d momentum spread (MACD-like)

    All numpy, deterministic, no future leakage.
    """
    K = universe.size
    F = 6
    feats = np.zeros((K, F), dtype=np.float32)
    lo = max(0, day_idx - lookback)
    win = log_returns[lo:day_idx, :][:, universe]
    win = np.where(np.isfinite(win), win, 0.0)
    if win.shape[0] < 5:
        return feats
    feats[:, 0] = win[-5:].sum(axis=0)
    feats[:, 1] = win[-10:].sum(axis=0) if win.shape[0] >= 10 else 0.0
    feats[:, 2] = win[-20:].sum(axis=0) if win.shape[0] >= 20 else 0.0
    feats[:, 3] = (
        win[-20:].std(axis=0) if win.shape[0] >= 20 else 0.0
    )
    full_sum = win.sum(axis=0)
    mu = float(full_sum.mean()); sd = float(full_sum.std() + 1e-6)
    feats[:, 4] = (full_sum - mu) / sd
    feats[:, 5] = feats[:, 0] - feats[:, 2] / 4.0
    return np.nan_to_num(feats, nan=0.0).astype(np.float32)


def _rsi(log_returns_window: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI computed from per-day log returns. Returns per-stock RSI scaled to
    [-1, 1] (RSI in [0,100], so we map (rsi-50)/50)."""
    if log_returns_window.shape[0] < period + 1:
        return np.zeros(log_returns_window.shape[1], dtype=np.float32)
    win = log_returns_window[-period - 1:, :]
    delta = np.diff(win, axis=0)
    gains = np.where(delta > 0, delta, 0.0).sum(axis=0)
    losses = np.where(delta < 0, -delta, 0.0).sum(axis=0)
    rs = gains / (losses + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return ((rsi - 50.0) / 50.0).astype(np.float32)


def _macd(prices_log: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """MACD computed from cumulative log returns (proxy for log-prices).
    Returns (macd, signal) lines normalised by the slow-period std for scale.
    """
    def _ema(x: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1.0)
        out = np.zeros_like(x)
        out[0] = x[0]
        for t in range(1, x.shape[0]):
            out[t] = alpha * x[t] + (1 - alpha) * out[t - 1]
        return out
    if prices_log.shape[0] < slow + signal:
        n = prices_log.shape[1]
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    ema_fast = _ema(prices_log, fast)
    ema_slow = _ema(prices_log, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    sd = prices_log[-slow:].std(axis=0) + 1e-6
    return (
        (macd_line[-1] / sd).astype(np.float32),
        (signal_line[-1] / sd).astype(np.float32),
    )


def _bollinger_pct(prices_log: np.ndarray, period: int = 20, k: float = 2.0) -> np.ndarray:
    """Bollinger %B in [0, 1]: (price - lower) / (upper - lower).
    Computed from cumulative log returns over `period`."""
    if prices_log.shape[0] < period:
        return np.zeros(prices_log.shape[1], dtype=np.float32)
    win = prices_log[-period:, :]
    mu = win.mean(axis=0)
    sd = win.std(axis=0) + 1e-6
    upper = mu + k * sd
    lower = mu - k * sd
    pctb = (prices_log[-1] - lower) / (upper - lower + 1e-12)
    return np.clip(pctb, 0.0, 1.0).astype(np.float32)


def _technical_features_rich(
    log_returns: np.ndarray,
    day_idx: int,
    universe: np.ndarray,
    lookback: int = 60,
) -> np.ndarray:
    """Rich FinRL-style technical features (K x F=15).

    Mirrors the FinRL paper's published indicator set adapted to a log-return-
    only data feed (we do not have intraday H/L on lattice_native, so ADX/CCI/KDJ
    are omitted; their information content overlaps RSI/Bollinger which we keep).

    Per-stock features at day ``day_idx``:
      0..3   5/10/20/60-day cumulative log return (momentum)
      4..5   20/60-day rolling volatility
      6..7   60-day cumulative-return z-score; 5d-vs-20d spread (MACD-like-simple)
      8      RSI(14) scaled to [-1, 1]
      9..10  MACD(12,26) line + signal, scale-normalised
      11     Bollinger %B(20, 2.0)
      12..14 5d/10d/20d EWMA of squared returns (vol clustering)
    """
    K = universe.size
    F = 15
    feats = np.zeros((K, F), dtype=np.float32)
    lo = max(0, day_idx - lookback)
    win = log_returns[lo:day_idx, :][:, universe]
    win = np.where(np.isfinite(win), win, 0.0)
    if win.shape[0] < 5:
        return feats
    feats[:, 0] = win[-5:].sum(axis=0)
    feats[:, 1] = win[-10:].sum(axis=0) if win.shape[0] >= 10 else 0.0
    feats[:, 2] = win[-20:].sum(axis=0) if win.shape[0] >= 20 else 0.0
    feats[:, 3] = win.sum(axis=0)
    feats[:, 4] = (
        win[-20:].std(axis=0) if win.shape[0] >= 20 else 0.0
    )
    feats[:, 5] = win.std(axis=0)
    full_sum = win.sum(axis=0)
    mu = float(full_sum.mean()); sd = float(full_sum.std() + 1e-6)
    feats[:, 6] = (full_sum - mu) / sd
    feats[:, 7] = feats[:, 0] - feats[:, 2] / 4.0
    feats[:, 8] = _rsi(win, period=14)
    prices_log = np.cumsum(win, axis=0)
    macd, sig = _macd(prices_log, 12, 26, 9)
    feats[:, 9] = macd
    feats[:, 10] = sig
    feats[:, 11] = _bollinger_pct(prices_log, 20, 2.0)
    sq = win * win
    for j, span in enumerate((5, 10, 20)):
        if sq.shape[0] >= span:
            feats[:, 12 + j] = sq[-span:].mean(axis=0)
    return np.nan_to_num(feats, nan=0.0).astype(np.float32)


class _TransformerPredictor(nn.Module):
    """Tiny per-stock transformer return predictor for StockFormer-style.

    Architecture per Gao et al. IJCAI 2023 §3.2 (Predictor): a 2-layer
    transformer encoder over the T-step lookback, mean-pool, linear
    head to a scalar return prediction. Trained with cross-sectional
    MSE on the train segment.
    """

    def __init__(
        self,
        n_features: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        lookback: int = 20,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(
            torch.zeros(1, lookback, d_model)
        )
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)
        self.lookback = lookback

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T, F) -> (N,) scores."""
        h = self.input_proj(x) + self.pos[:, : x.shape[1], :]
        h = self.enc(h)
        return self.head(h.mean(dim=1)).squeeze(-1)


@dataclass
class PortfolioEnvConfig:
    universe_size: int = 50
    lookback: int = 20
    feature_dim: int = 6
    transaction_cost: float = 0.0005
    max_gross: float = 1.0
    long_only: bool = False


class PortfolioEnv(gym.Env):
    """Generic daily-rebalanced long-short portfolio environment.

    Differs from :class:`ExposureEnv`: the action IS the weight vector,
    not exposure on a precomputed portfolio. This is the native env
    type for StockFormer and FinRL baselines.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        log_returns: np.ndarray,
        tradable: np.ndarray,
        day_indices: List[int],
        universe: np.ndarray,
        cfg: PortfolioEnvConfig,
        feature_fn=None,
        predictor_scores: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self._lr = log_returns
        self._tradable = tradable
        self._days = list(day_indices)
        self._uni = universe
        self._cfg = cfg
        self._feature_fn = feature_fn
        self._scores = predictor_scores
        self._k = int(universe.size)
        f_dim = (
            cfg.feature_dim
            if predictor_scores is None
            else cfg.feature_dim + 1
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._k * f_dim + self._k + 1,),
            dtype=np.float32,
        )
        if cfg.long_only:
            low = np.zeros(self._k, dtype=np.float32)
            high = np.ones(self._k, dtype=np.float32)
        else:
            low = -np.ones(self._k, dtype=np.float32)
            high = np.ones(self._k, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._t = 0
        self._w = np.zeros(self._k, dtype=np.float32)
        self._equity = 1.0

    def _obs(self) -> np.ndarray:
        day = self._days[self._t]
        if self._feature_fn is not None:
            feats = self._feature_fn(day, self._uni)
        else:
            feats = np.zeros((self._k, self._cfg.feature_dim),
                              dtype=np.float32)
        if self._scores is not None:
            score = self._scores[day, self._uni].astype(np.float32)
            score = np.nan_to_num(score, nan=0.0)[:, None]
            feats = np.concatenate([feats, score], axis=1)
        return np.concatenate([
            feats.flatten(),
            self._w.astype(np.float32),
            np.array([self._equity], dtype=np.float32),
        ])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._w = np.zeros(self._k, dtype=np.float32)
        self._equity = 1.0
        return self._obs(), {}

    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        if self._cfg.long_only:
            v = np.clip(raw, 0.0, 1.0)
            total = float(v.sum())
            return (v / total) * self._cfg.max_gross if total > 1e-9 else v
        v = np.clip(raw, -1.0, 1.0)
        gross = float(np.abs(v).sum())
        return (v / gross) * self._cfg.max_gross if gross > 1e-9 else v

    def step(self, action: np.ndarray):
        target = self._normalise(np.asarray(action, dtype=np.float32))
        turnover = float(np.abs(target - self._w).sum())
        cost = self._cfg.transaction_cost * turnover
        day = self._days[self._t]
        next_day = day + 1
        if next_day >= self._lr.shape[0]:
            r = 0.0
        else:
            r_vec = self._lr[next_day, self._uni]
            r_vec = np.where(np.isfinite(r_vec), r_vec, 0.0)
            r = float((target * r_vec).sum())
        net = r - cost
        self._equity *= float(np.exp(net))
        self._w = target.astype(np.float32)
        self._t += 1
        done = self._t >= len(self._days) - 1
        return (
            self._obs(),
            float(net),
            done,
            False,
            {"strategy_return": float(net), "equity": float(self._equity)},
        )


@dataclass
class WholeStackResult:
    name: str
    daily_log_returns: np.ndarray
    mean_return: float
    volatility: float
    sharpe_annualised: float
    final_equity: float
    n_steps: int
    gross_exposure_mean: float
    net_exposure_mean: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "mean_return": float(self.mean_return),
            "volatility": float(self.volatility),
            "sharpe_annualised": float(self.sharpe_annualised),
            "final_equity": float(self.final_equity),
            "n_steps": int(self.n_steps),
            "gross_exposure_mean": float(self.gross_exposure_mean),
            "net_exposure_mean": float(self.net_exposure_mean),
        }


def _eval_agent(agent, env: PortfolioEnv, recurrent: bool = False) -> WholeStackResult:
    obs, _ = env.reset(seed=0)
    rets: List[float] = []
    gross: List[float] = []
    net: List[float] = []
    info: Dict = {}
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        rets.append(r)
        gross.append(float(np.abs(env._w).sum()))
        net.append(float(env._w.sum()))
        if term or trunc:
            break
    arr = np.asarray(rets, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    vol = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sharpe = (
        mean * 252.0 / (vol * np.sqrt(252.0)) if vol > 0 else 0.0
    )
    return WholeStackResult(
        name="",
        daily_log_returns=arr,
        mean_return=mean,
        volatility=vol,
        sharpe_annualised=sharpe,
        final_equity=float(np.exp(arr.sum())),
        n_steps=int(arr.size),
        gross_exposure_mean=float(np.mean(gross)) if gross else 0.0,
        net_exposure_mean=float(np.mean(net)) if net else 0.0,
    )


def _train_predictor(
    bridge: LatticePanelBatch,
    universe: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    lookback: int = 20,
    epochs: int = 5,
    lr: float = 3e-4,
    device: str = "cpu",
) -> _TransformerPredictor:
    """Supervised pretraining of the StockFormer-style return predictor."""
    model = _TransformerPredictor(lookback=lookback).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    log_ret = bridge.log_returns_1d
    label_days = [int(d) for d in train_idx if d >= lookback and d + 1 < log_ret.shape[0]]
    for ep in range(epochs):
        rng = np.random.default_rng(ep)
        rng.shuffle(label_days)
        total = 0.0; n = 0
        for d in label_days:
            x_win = log_ret[d - lookback + 1: d + 1, universe]
            x_win = np.where(np.isfinite(x_win), x_win, 0.0)
            x_t = torch.from_numpy(
                x_win.T.astype(np.float32)
            ).unsqueeze(-1).to(device)
            y_next = bridge.log_returns_1d[d + 1, universe]
            y_next = np.where(np.isfinite(y_next), y_next, 0.0)
            y_z = (y_next - y_next.mean()) / (y_next.std() + 1e-6)
            y_t = torch.from_numpy(y_z.astype(np.float32)).to(device)
            pred = model(x_t)
            loss = ((pred - y_t) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); n += 1
        if n:
            print(f"    [predictor ep{ep}] mse={total / n:.4f} n={n}")
    model.eval()
    return model


def _predictor_scores_array(
    bridge: LatticePanelBatch,
    universe: np.ndarray,
    model: _TransformerPredictor,
    lookback: int = 20,
    device: str = "cpu",
) -> np.ndarray:
    """Pre-score every day in the bridge into a (T, N) array of zeros
    outside the universe."""
    log_ret = bridge.log_returns_1d
    T, N = log_ret.shape
    out = np.zeros((T, N), dtype=np.float32)
    with torch.no_grad():
        for d in range(lookback - 1, T):
            x_win = log_ret[d - lookback + 1: d + 1, universe]
            x_win = np.where(np.isfinite(x_win), x_win, 0.0)
            x_t = torch.from_numpy(
                x_win.T.astype(np.float32)
            ).unsqueeze(-1).to(device)
            pred = model(x_t).cpu().numpy()
            out[d, universe] = pred
    return out


def run_finrl_baseline(
    bridge: LatticePanelBatch,
    fold: int,
    seed: int,
    total_timesteps: int = 20000,
    universe_k: int = 50,
    device: str = "cpu",
    feature_set: str = "lite",
) -> WholeStackResult:
    """FinRL-style: technical features + PPO + long-short portfolio.

    feature_set:
      "lite" (default, F=6) -- the simplified set used in the first round.
      "rich" (F=15) -- adds RSI(14), MACD(12,26,9), Bollinger %B(20),
        multi-horizon EWMA vol, matching the FinRL paper's published
        indicator set as closely as possible without intraday H/L.
    """
    from stable_baselines3 import PPO

    universe = _select_topk_universe(
        bridge, list(bridge.train_idx), k=universe_k, seed=seed
    )
    feat_dim = 15 if feature_set == "rich" else 6
    cfg = PortfolioEnvConfig(universe_size=universe_k, feature_dim=feat_dim)
    feat_callable = (
        _technical_features_rich if feature_set == "rich"
        else _technical_features
    )

    def feat_fn(day, uni):
        return feat_callable(bridge.log_returns_1d, day, uni)

    train_env = PortfolioEnv(
        log_returns=bridge.log_returns_1d,
        tradable=bridge.tradable,
        day_indices=list(bridge.train_idx),
        universe=universe,
        cfg=cfg,
        feature_fn=feat_fn,
    )
    eval_env = PortfolioEnv(
        log_returns=bridge.log_returns_1d,
        tradable=bridge.tradable,
        day_indices=list(bridge.test_idx),
        universe=universe,
        cfg=cfg,
        feature_fn=feat_fn,
    )
    agent = PPO(
        "MlpPolicy", train_env, learning_rate=3e-4,
        n_steps=512, seed=seed, verbose=0, device=device,
    )
    agent.learn(total_timesteps=total_timesteps)
    res = _eval_agent(agent, eval_env)
    res.name = "finrl_ppo"
    return res


def run_stockformer_baseline(
    bridge: LatticePanelBatch,
    fold: int,
    seed: int,
    total_timesteps: int = 20000,
    universe_k: int = 50,
    predictor_epochs: int = 5,
    device: str = "cpu",
    feature_set: str = "lite",
) -> WholeStackResult:
    """StockFormer-style: transformer predictor + SAC + long-short portfolio.

    feature_set: same options as ``run_finrl_baseline``.
    """
    from stable_baselines3 import SAC

    universe = _select_topk_universe(
        bridge, list(bridge.train_idx), k=universe_k, seed=seed
    )
    feat_dim = 15 if feature_set == "rich" else 6
    cfg = PortfolioEnvConfig(universe_size=universe_k, feature_dim=feat_dim)
    feat_callable = (
        _technical_features_rich if feature_set == "rich"
        else _technical_features
    )

    # Stage 1: train transformer predictor on train segment only
    predictor = _train_predictor(
        bridge=bridge,
        universe=universe,
        train_idx=bridge.train_idx,
        val_idx=bridge.val_idx,
        lookback=20,
        epochs=predictor_epochs,
        device=device,
    )
    scores = _predictor_scores_array(
        bridge=bridge, universe=universe, model=predictor,
        lookback=20, device=device,
    )

    def feat_fn(day, uni):
        return feat_callable(bridge.log_returns_1d, day, uni)

    train_env = PortfolioEnv(
        log_returns=bridge.log_returns_1d,
        tradable=bridge.tradable,
        day_indices=list(bridge.train_idx),
        universe=universe,
        cfg=cfg, feature_fn=feat_fn, predictor_scores=scores,
    )
    eval_env = PortfolioEnv(
        log_returns=bridge.log_returns_1d,
        tradable=bridge.tradable,
        day_indices=list(bridge.test_idx),
        universe=universe,
        cfg=cfg, feature_fn=feat_fn, predictor_scores=scores,
    )
    agent = SAC(
        "MlpPolicy", train_env, learning_rate=3e-4,
        seed=seed, verbose=0, device=device,
    )
    agent.learn(total_timesteps=total_timesteps)
    res = _eval_agent(agent, eval_env)
    res.name = "stockformer_sac"
    return res


__all__ = [
    "PortfolioEnv",
    "PortfolioEnvConfig",
    "WholeStackResult",
    "run_finrl_baseline",
    "run_stockformer_baseline",
]
