"""DeepTrader baseline eval: universal panel + DJIA-30 credibility gate.

Two phases, analogous to ``finrl_faithful_eval`` and
``stockformer_faithful_eval``:

Phase ``universal``: top-30 most-active tickers per fold from the
lattice_native S&P 500 panel under the 5-fold macro-stratified InVAR-RL
protocol. The 6 ASU per-stock features are subsetted from the bridge
``x`` tensor; the 4 MSU market features (VIX, term-structure spread,
credit spread, cross-sectional dispersion) are subsetted from
``bridge.macro_arr`` plus ``bridge.cs_disp_z``. The 550 x 550 GICS sector
adjacency is sliced down to the chosen 30 tickers.

Phase ``djia``: train 2009-2015, test 2016-2020 on DJIA-30 via yfinance,
matching the FinRL credibility window. Reported as a sanity check in
the paper.

Usage::

    # Universal phase (one cell)
    python -m invar_rl.training.deeptrader_eval \\
        --phase universal --fold 1 --seed 42 --total-epochs 500

    # DJIA credibility (one seed)
    python -m invar_rl.training.deeptrader_eval \\
        --phase djia --seed 42 --total-epochs 500
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from invar_rl.baselines.deeptrader import (
    DeepTraderActor,
    DeepTraderEnv,
    DeepTraderEnvConfig,
    DeepTraderTrainConfig,
    evaluate_deeptrader,
    train_deeptrader,
)


# --------------------------------------------------------------------- #
# Feature builders
# --------------------------------------------------------------------- #


# Indices into the lattice_native 26-col per-stock feature vector. See
# ``src/v2/data/lattice_native_panel.py`` for the exact column order.
# We subset 6 features as a paper-faithful surrogate for DeepTrader's
# (close-return, log-volume z, ATR/close, RSI/100, MACD signal,
# 5-day momentum). Bridge ``x`` is already train-standardised so these
# are z-scored.
_ASU_FEATURE_INDICES: Tuple[int, ...] = (
    0,  # log_return            -> close-return
    3,  # log_volume            -> log-volume z-score
    7,  # high_low_range        -> ATR / close proxy
    8,  # close_to_high_5d      -> RSI / 100 proxy (mean-reversion)
    1,  # log_return_5d         -> MACD signal proxy (short trend)
    2,  # log_return_20d        -> 5-day momentum proxy (medium trend)
)


# Macro column names in ``bridge.macro_arr``. The bridge does not expose
# ``macro_cols`` so we recompute them here from the canonical
# ``MACRO_FEATURE_COLS_FULL`` constant intersected with the on-disk
# parquet columns; in practice this is the full 28-col list for
# lattice_native panels.
_MSU_MACRO_COLS: Tuple[str, ...] = (
    "vix",          # VIX
    "term_10y_2y",  # term-structure spread
    "hy_spread",    # credit spread
)


def _resolve_macro_indices(panel_kind: str) -> List[int]:
    """Return the column indices of the MSU macro features in macro_arr.

    Re-reads the canonical macro parquet, intersects with the global
    ``MACRO_FEATURE_COLS_FULL`` ordering used inside
    ``standardize_macro_duration``, and returns positional indices for
    the columns in ``_MSU_MACRO_COLS``.

    Args:
        panel_kind: Either ``"lattice_native"`` or ``"biotech"``.

    Returns:
        List of positional indices into ``bridge.macro_arr``.
    """
    from src.v2.data.macro_duration_features import MACRO_FEATURE_COLS_FULL
    if panel_kind == "lattice_native":
        parquet_path = Path("data/processed/macro_duration_features_sp500.parquet")
    else:
        parquet_path = Path("data/processed/macro_duration_features.parquet")
    cols_on_disk = list(pd.read_parquet(parquet_path).columns)
    ordered = [c for c in MACRO_FEATURE_COLS_FULL if c in cols_on_disk]
    return [ordered.index(c) for c in _MSU_MACRO_COLS]


def _slice_sector_adjacency(tickers: List[str]) -> np.ndarray:
    """Slice the 550 x 550 GICS sector adjacency down to ``tickers``.

    Tickers absent from the saved sector map are placed in a synthetic
    "Unknown" group and connected only to themselves, which mirrors the
    fallback in :mod:`invar_rl.scripts.build_sector_adjacency`.

    Args:
        tickers: Sub-universe of ticker symbols, length K.

    Returns:
        ``(K, K)`` float32 ndarray with 1.0 on same-sector pairs and on
        the diagonal, 0.0 elsewhere.
    """
    sm = pd.read_csv("data/processed/sp500_sector_map.csv")
    full_order: List[str] = sm["ticker"].tolist()
    full_adj = np.load("data/processed/sp500_sector_adjacency.npy")
    full_sector = dict(zip(sm["ticker"], sm["sector"]))
    name_to_idx = {t: i for i, t in enumerate(full_order)}

    k = len(tickers)
    sub = np.zeros((k, k), dtype=np.float32)
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if ti in name_to_idx and tj in name_to_idx:
                sub[i, j] = float(full_adj[name_to_idx[ti], name_to_idx[tj]])
            else:
                # Fallback: same-sector via the map, else self-loop only.
                si = full_sector.get(ti, f"_unknown_{ti}")
                sj = full_sector.get(tj, f"_unknown_{tj}")
                sub[i, j] = 1.0 if si == sj else 0.0
    np.fill_diagonal(sub, 1.0)
    return sub


def _build_asu_features(bridge_x: torch.Tensor, universe: np.ndarray) -> np.ndarray:
    """Subset ASU's 6 per-stock features from the bridge ``x`` tensor.

    Args:
        bridge_x: Standardised feature tensor, shape ``(T, N, F)``.
        universe: Asset indices into ``N``, length K.

    Returns:
        ``(T, K, 6)`` float32 ndarray of standardised ASU features.
    """
    x = bridge_x.detach().cpu().numpy()
    x = x[:, universe, :]
    x = x[..., list(_ASU_FEATURE_INDICES)]
    return x.astype(np.float32)


def _build_msu_features(
    bridge_macro_arr: np.ndarray,
    bridge_cs_disp_z: np.ndarray,
    panel_kind: str,
) -> np.ndarray:
    """Build the 4 MSU market features for the full panel duration.

    Concatenates VIX, term-structure spread, credit spread, and
    cross-sectional dispersion. All four are already z-scored against
    the train fold by ``standardize_macro_duration`` and
    ``build_lattice_bridge``.

    Args:
        bridge_macro_arr: Macro tensor, shape ``(T, macro_dim)``.
        bridge_cs_disp_z: Cross-sectional dispersion z-score, shape ``(T,)``.
        panel_kind: Either ``"lattice_native"`` or ``"biotech"``.

    Returns:
        ``(T, 4)`` float32 ndarray of MSU market features.
    """
    indices = _resolve_macro_indices(panel_kind)
    macro_subset = bridge_macro_arr[:, indices]
    out = np.concatenate(
        [macro_subset, bridge_cs_disp_z[:, None]], axis=1,
    ).astype(np.float32)
    return out


def _segment_returns(bridge_log_returns: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """Slice next-day returns for the sub-universe.

    DeepTraderEnv expects a ``(T, K)`` ndarray of next-day per-asset
    returns. We pass the simple-return approximation ``exp(log_ret) - 1``
    so REINFORCE rewards are interpretable as portfolio P&L.

    Args:
        bridge_log_returns: Bridge ``log_returns_1d`` tensor, shape ``(T, N)``.
        universe: Asset indices, length K.

    Returns:
        ``(T, K)`` float32 ndarray.
    """
    lr = bridge_log_returns[:, universe].astype(np.float64)
    lr = np.where(np.isfinite(lr), lr, 0.0)
    return (np.exp(lr) - 1.0).astype(np.float32)


# --------------------------------------------------------------------- #
# Universal phase
# --------------------------------------------------------------------- #


def run_universal_phase(
    fold: int,
    seed: int,
    output_dir: Path,
    total_epochs: int = 500,
    universe_k: int = 30,
    rollout_steps: int = 12,
    batch_size: int = 37,
) -> dict:
    """Universal phase: top-K most-active tickers from the lattice_native panel.

    Args:
        fold: Fold index in ``{1, 2, 3, 4, 5}``.
        seed: Random seed.
        output_dir: Directory to write the result JSON to.
        total_epochs: Number of REINFORCE epochs.
        universe_k: Sub-universe size, default 30.
        rollout_steps: REINFORCE trajectory length per gradient step.
        batch_size: Trajectories per epoch.

    Returns:
        Payload dict mirroring the JSON written to disk.
    """
    from src.invar import InVARConfig
    from invar_rl.data.lattice_bridge import build_lattice_bridge

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg_inv = InVARConfig(fold=fold, seed=seed)
    cfg_inv.panel_kind = "lattice_native"
    cfg_inv.two_regime_val = True
    cfg_inv.panel_end = "2025-12-31"
    bridge = build_lattice_bridge(cfg_inv)

    # Top-K most-active tickers in the train segment (same rule as the
    # FinRL/StockFormer universal phases).
    train_active = bridge.tradable[bridge.train_idx].sum(axis=0)
    order = np.argsort(-train_active)
    universe_idx = np.sort(order[:universe_k])
    tickers = [bridge.tickers[int(i)] for i in universe_idx]

    print(
        f"[deeptrader universal] fold={fold} seed={seed} "
        f"K={universe_k} train_n={len(bridge.train_idx)} "
        f"test_n={len(bridge.test_idx)}"
    )

    # Build inputs.
    asu_features = _build_asu_features(bridge.x, universe_idx)
    msu_features = _build_msu_features(
        bridge.macro_arr, bridge.cs_disp_z, panel_kind="lattice_native",
    )
    returns = _segment_returns(bridge.log_returns_1d, universe_idx)
    adjacency = _slice_sector_adjacency(tickers)

    # Slice each (returns, ASU, MSU) array into train and test segments
    # by the bridge's train_idx / test_idx so the env never sees the
    # val segment. We re-anchor each segment to start at day 0.
    def _gather(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return arr[idx].copy()

    train_returns = _gather(returns, bridge.train_idx)
    train_stocks = _gather(asu_features, bridge.train_idx)
    train_market = _gather(msu_features, bridge.train_idx)
    test_returns = _gather(returns, bridge.test_idx)
    test_stocks = _gather(asu_features, bridge.test_idx)
    test_market = _gather(msu_features, bridge.test_idx)
    val_returns = _gather(returns, bridge.val_idx)
    val_stocks = _gather(asu_features, bridge.val_idx)
    val_market = _gather(msu_features, bridge.val_idx)

    # Upstream-faithful env: monthly rebalance (trade_len=21),
    # 10 bps fee, 12-month episode cap. window=60 daily bars is our
    # paper-equivalent substitute for upstream's 13 weekly bars
    # (5 * (13 + 1) = 70 calendar days).
    env_cfg_train = DeepTraderEnvConfig(
        window=60, fee_bps=10.0, trade_len=21, episode_length=12,
    )
    env_cfg_eval = DeepTraderEnvConfig(
        window=60, fee_bps=10.0, trade_len=21, episode_length=None,
    )
    train_env = DeepTraderEnv(
        returns=train_returns,
        stock_features=train_stocks,
        market_features=train_market,
        config=env_cfg_train,
    )
    val_env = DeepTraderEnv(
        returns=val_returns,
        stock_features=val_stocks,
        market_features=val_market,
        config=env_cfg_eval,
    )
    test_env = DeepTraderEnv(
        returns=test_returns,
        stock_features=test_stocks,
        market_features=test_market,
        config=env_cfg_eval,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Upstream G=4 for DJIA-30. For the universal K=30 panel we keep
    # G=4 to match (long 4, short 4 out of 30).
    actor = DeepTraderActor(
        num_assets=universe_k,
        top_g=4,
        asu_kwargs={"in_features": len(_ASU_FEATURE_INDICES)},
        msu_kwargs={"in_features": 4},
    )
    cfg_train = DeepTraderTrainConfig(
        epochs=total_epochs,
        lr=1e-6,
        weight_decay=1e-3,
        gamma=0.05,
        batch_size=batch_size,
        rollout_steps=rollout_steps,
        grad_clip=100.0,
        eval_every=max(1, total_epochs // 10),
    )

    print(f"[deeptrader universal] training {total_epochs} epochs on {device}")
    best_state, losses = train_deeptrader(
        actor, train_env,
        adjacency=adjacency,
        device=device,
        val_env=val_env,
        config=cfg_train,
        verbose=True,
    )
    actor.load_state_dict(best_state)
    perf = evaluate_deeptrader(actor, test_env, adjacency, device=device)
    perf["fold"] = fold
    perf["seed"] = seed
    print(
        f"[deeptrader universal] fold={fold} seed={seed} "
        f"sharpe={perf['sharpe']:+.3f} ann_ret={perf['ann_return']:+.4f} "
        f"eq={perf['final_equity']:.4f}"
    )

    payload = {
        "phase": "universal",
        "fold": fold,
        "seed": seed,
        "universe_k": universe_k,
        "n_train_days": int(len(bridge.train_idx)),
        "n_val_days": int(len(bridge.val_idx)),
        "n_test_days": int(len(bridge.test_idx)),
        "tickers": tickers,
        "perf": perf,
        "config": {
            "total_epochs": total_epochs,
            "lr": cfg_train.lr,
            "gamma": cfg_train.gamma,
            "batch_size": cfg_train.batch_size,
            "rollout_steps": cfg_train.rollout_steps,
            "window": env_cfg_train.window,
            "fee_bps": env_cfg_train.fee_bps,
            "trade_len": env_cfg_train.trade_len,
            "top_g": 4,
            "asu_in_features": len(_ASU_FEATURE_INDICES),
            "msu_in_features": 4,
        },
        "loss_history_tail": losses[-min(20, len(losses)):],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[deeptrader universal] wrote {out_path}")
    return payload


# --------------------------------------------------------------------- #
# DJIA credibility phase
# --------------------------------------------------------------------- #


def _fetch_djia_panel(
    tickers: Tuple[str, ...],
    start: str,
    end: str,
    min_coverage: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
    """Download DJIA-30 OHLCV via yfinance and assemble a per-stock panel.

    Args:
        tickers: Candidate DJIA-30 tickers.
        start: Start date string.
        end: End date string.
        min_coverage: Minimum fraction of union trading days a ticker
            must cover to stay in the panel.

    Returns:
        Tuple ``(close, log_returns, volume, dates, kept_tickers)`` where
        ``close`` and ``volume`` are ``(T, N)`` float64, ``log_returns``
        is ``(T, N)`` float64 with zeros at t=0, and ``dates`` is a list
        of ``T`` trading-day timestamps.
    """
    import yfinance as yf
    raw_dfs: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = yf.download(
                t, start=start, end=end, auto_adjust=True,
                progress=False, threads=False,
            )
        except Exception:
            continue
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df = df.set_index("date")
        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(df.columns):
            continue
        raw_dfs[t] = df[["open", "high", "low", "close", "volume"]]
    if not raw_dfs:
        raise RuntimeError("yfinance returned no tickers for DJIA panel")
    union_index = sorted(set().union(*[df.index for df in raw_dfs.values()]))
    union_index = pd.DatetimeIndex(union_index)
    T = len(union_index)
    kept = sorted([
        t for t, df in raw_dfs.items()
        if len(df.index.intersection(union_index)) / T >= min_coverage
    ])
    if not kept:
        raise RuntimeError("no DJIA tickers met coverage threshold")
    close = np.zeros((T, len(kept)), dtype=np.float64)
    volume = np.zeros((T, len(kept)), dtype=np.float64)
    high = np.zeros((T, len(kept)), dtype=np.float64)
    low = np.zeros((T, len(kept)), dtype=np.float64)
    for j, t in enumerate(kept):
        df = raw_dfs[t].reindex(union_index).ffill().bfill()
        close[:, j] = df["close"].fillna(0.0).values
        volume[:, j] = df["volume"].fillna(0.0).values
        high[:, j] = df["high"].fillna(0.0).values
        low[:, j] = df["low"].fillna(0.0).values
    log_returns = np.zeros_like(close, dtype=np.float64)
    log_returns[1:] = np.log(
        np.clip(close[1:] / np.clip(close[:-1], 1e-6, None), 1e-6, None)
    )
    return close, log_returns, volume, list(union_index), kept


def _djia_asu_features(
    close: np.ndarray, log_returns: np.ndarray, volume: np.ndarray,
) -> np.ndarray:
    """Build a 6-feature ASU input from raw DJIA OHLCV.

    Args:
        close: ``(T, N)`` close prices.
        log_returns: ``(T, N)`` log returns.
        volume: ``(T, N)`` traded volume.

    Returns:
        ``(T, N, 6)`` float32 z-scored feature ndarray.
    """
    T, N = close.shape
    feats = np.zeros((T, N, 6), dtype=np.float64)
    feats[..., 0] = log_returns  # close-return
    log_vol = np.log(np.clip(volume, 1.0, None))
    feats[..., 1] = log_vol  # log-volume
    # ATR / close proxy: 20-day rolling std of |log_returns|.
    abs_ret = np.abs(log_returns)
    for n in range(N):
        s = pd.Series(abs_ret[:, n]).rolling(20, min_periods=5).mean().to_numpy()
        feats[:, n, 2] = np.nan_to_num(s, nan=0.0)
    # 5-day momentum.
    feats[5:, :, 5] = log_returns[5:] - log_returns[:-5]
    # MACD signal: 12-day EMA - 26-day EMA of close.
    for n in range(N):
        s = pd.Series(close[:, n])
        macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
        feats[:, n, 4] = np.nan_to_num(macd.to_numpy(), nan=0.0)
    # RSI / 100.
    for n in range(N):
        delta = np.diff(close[:, n], prepend=close[0, n])
        gain = np.clip(delta, 0.0, None)
        loss = np.clip(-delta, 0.0, None)
        avg_g = pd.Series(gain).rolling(14, min_periods=5).mean()
        avg_l = pd.Series(loss).rolling(14, min_periods=5).mean()
        rs = avg_g / avg_l.replace(0.0, 1e-6)
        rsi = (100.0 - 100.0 / (1.0 + rs)) / 100.0
        feats[:, n, 3] = np.nan_to_num(rsi.to_numpy(), nan=0.5)

    # Cross-sectional z-score per feature per day.
    out = feats.copy()
    for k in range(6):
        col = out[..., k]
        mu = np.nanmean(col, axis=1, keepdims=True)
        sd = np.nanstd(col, axis=1, keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)
        out[..., k] = (col - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _djia_msu_features(close: np.ndarray) -> np.ndarray:
    """Build 4 MSU features from the DJIA panel itself (no external macros).

    Constructs a synthetic 4-feature market vector via:
    market log-return, market 20-day vol, market range / close,
    cross-sectional dispersion. This is the simplest paper-equivalent
    that can be assembled without fetching the FRED macros for the DJIA
    credibility gate.

    Args:
        close: ``(T, N)`` close prices.

    Returns:
        ``(T, 4)`` float32 ndarray.
    """
    T = close.shape[0]
    market = close.mean(axis=1)
    log_ret = np.zeros(T, dtype=np.float64)
    log_ret[1:] = np.log(np.clip(market[1:] / np.clip(market[:-1], 1e-6, None), 1e-6, None))
    vol = pd.Series(log_ret).rolling(20, min_periods=5).std().to_numpy()
    rng = (close.max(axis=1) - close.min(axis=1)) / np.clip(market, 1e-6, None)
    cs_disp = close.std(axis=1) / np.clip(market, 1e-6, None)
    out = np.stack([log_ret, vol, rng, cs_disp], axis=1)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    mu = out.mean(axis=0, keepdims=True)
    sd = out.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((out - mu) / sd).astype(np.float32)


def run_djia_credibility(
    seed: int,
    output_dir: Path,
    total_epochs: int = 500,
    train_end: str = "2015-12-31",
    test_start: str = "2016-01-04",
    test_end: str = "2020-06-30",
    rollout_steps: int = 12,
    batch_size: int = 37,
) -> dict:
    """DJIA-30 credibility phase, train 2009-2015 / test 2016-2020.

    Args:
        seed: Random seed.
        output_dir: Directory to write the result JSON to.
        total_epochs: REINFORCE epoch budget.
        train_end: Last train date (inclusive).
        test_start: First test date (inclusive).
        test_end: Last test date (inclusive).
        rollout_steps: REINFORCE trajectory length.
        batch_size: Trajectories per epoch.

    Returns:
        Payload dict mirroring the JSON written to disk.
    """
    from invar_rl.baselines.finrl_faithful import DJIA_30_TICKERS

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[deeptrader djia] fetching DJIA-30 prices via yfinance...")
    close, log_returns, volume, dates, tickers = _fetch_djia_panel(
        DJIA_30_TICKERS, start="2009-01-01", end=test_end,
    )
    print(f"[deeptrader djia] kept {len(tickers)} tickers; T={len(dates)}")

    train_end_idx = max(
        i for i, d in enumerate(dates) if d <= pd.Timestamp(train_end)
    )
    test_start_idx = min(
        i for i, d in enumerate(dates) if d >= pd.Timestamp(test_start)
    )

    # Returns as simple per-day.
    simple_returns = (np.exp(log_returns) - 1.0).astype(np.float32)
    stock_feats = _djia_asu_features(close, log_returns, volume)
    market_feats = _djia_msu_features(close)
    adjacency = _slice_sector_adjacency(tickers)

    # Upstream-faithful env: monthly rebalance, 10 bps fee, 12-month cap.
    env_cfg_train = DeepTraderEnvConfig(
        window=60, fee_bps=10.0, trade_len=21, episode_length=12,
    )
    env_cfg_eval = DeepTraderEnvConfig(
        window=60, fee_bps=10.0, trade_len=21, episode_length=None,
    )
    train_env = DeepTraderEnv(
        returns=simple_returns[: train_end_idx + 1],
        stock_features=stock_feats[: train_end_idx + 1],
        market_features=market_feats[: train_end_idx + 1],
        config=env_cfg_train,
    )
    test_env = DeepTraderEnv(
        returns=simple_returns[test_start_idx:],
        stock_features=stock_feats[test_start_idx:],
        market_features=market_feats[test_start_idx:],
        config=env_cfg_eval,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Upstream G=4 for DJIA-30 (hyper.json).
    actor = DeepTraderActor(
        num_assets=len(tickers),
        top_g=4,
        asu_kwargs={"in_features": 6},
        msu_kwargs={"in_features": 4},
    )
    cfg_train = DeepTraderTrainConfig(
        epochs=total_epochs,
        lr=1e-6,
        weight_decay=1e-3,
        gamma=0.05,
        batch_size=batch_size,
        rollout_steps=rollout_steps,
        grad_clip=100.0,
        eval_every=max(1, total_epochs // 10),
    )

    print(f"[deeptrader djia] training {total_epochs} epochs on {device}")
    best_state, losses = train_deeptrader(
        actor, train_env,
        adjacency=adjacency,
        device=device,
        val_env=None,
        config=cfg_train,
        verbose=True,
    )
    actor.load_state_dict(best_state)
    perf = evaluate_deeptrader(actor, test_env, adjacency, device=device)
    perf["seed"] = seed
    print(
        f"[deeptrader djia] seed={seed} sharpe={perf['sharpe']:+.3f} "
        f"ann_ret={perf['ann_return']:+.4f} eq={perf['final_equity']:.4f}"
    )

    payload = {
        "phase": "djia",
        "seed": seed,
        "n_tickers": len(tickers),
        "n_train_days": int(train_end_idx + 1),
        "n_test_days": int(len(dates) - test_start_idx),
        "tickers": tickers,
        "perf": perf,
        "config": {
            "total_epochs": total_epochs,
            "lr": cfg_train.lr,
            "gamma": cfg_train.gamma,
            "batch_size": cfg_train.batch_size,
            "rollout_steps": cfg_train.rollout_steps,
            "window": env_cfg_train.window,
            "fee_bps": env_cfg_train.fee_bps,
            "trade_len": env_cfg_train.trade_len,
            "top_g": 4,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        },
        "loss_history_tail": losses[-min(20, len(losses)):],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"djia_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[deeptrader djia] wrote {out_path}")
    return payload


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="DeepTrader baseline eval.")
    p.add_argument(
        "--phase", type=str, required=True, choices=["universal", "djia"],
    )
    p.add_argument("--fold", type=int, default=None, choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir-root", type=str,
        default="invar_rl/results/deeptrader",
    )
    p.add_argument("--total-epochs", type=int, default=500)
    p.add_argument("--universe-k", type=int, default=30)
    p.add_argument("--rollout-steps", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=37)
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    out_dir = Path(args.output_dir_root) / args.phase
    if args.phase == "universal":
        if args.fold is None:
            raise SystemExit("--fold is required for universal phase")
        run_universal_phase(
            fold=args.fold, seed=args.seed,
            output_dir=out_dir,
            total_epochs=args.total_epochs,
            universe_k=args.universe_k,
            rollout_steps=args.rollout_steps,
            batch_size=args.batch_size,
        )
    else:
        run_djia_credibility(
            seed=args.seed,
            output_dir=out_dir,
            total_epochs=args.total_epochs,
            rollout_steps=args.rollout_steps,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
