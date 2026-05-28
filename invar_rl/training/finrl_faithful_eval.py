"""Faithful FinRL eval: reproduce on DJIA-30 + run on universal protocol.

Two phases:

Phase 1 (credibility gate): train PPO + A2C + DDPG on DJIA-30, train
2009-01-01 to 2015-12-31, test 2016-01-04 to 2020-06-30 (the FinRL
paper's reference window). Target ensemble Sharpe ~1.30.

Phase 2 (stress test on our universe): re-run faithful FinRL on the
universal S&P 500 lattice_native panel under our 5-fold macro-
stratified protocol. This is the apples-to-apples comparison with
\\sysname.

Usage::

    # Phase 1: credibility gate
    python -m invar_rl.training.finrl_faithful_eval \\
        --phase djia --seed 42 --method ppo

    # Phase 2: our universe
    python -m invar_rl.training.finrl_faithful_eval \\
        --phase universal --fold 1 --seed 42 --method ppo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from invar_rl.baselines.finrl_faithful import (
    DJIA_30_TICKERS,
    FINRL_INDICATORS,
    FinRLEnvConfig,
    FinRLStockTradingEnv,
    _fetch_vix,
    _technical_indicators,
    _turbulence_index,
    evaluate_finrl_env,
    train_finrl_a2c,
    train_finrl_ddpg,
    train_finrl_ppo,
)


def _fetch_djia_prices(
    start: str = "2009-01-01",
    end: str = "2020-06-30",
) -> pd.DataFrame:
    """Download DJIA-30 prices via yfinance.

    Returns long-format with columns ticker, date, open, high, low,
    close, volume. Drops tickers with insufficient history.
    """
    import yfinance as yf
    tickers = sorted(set(DJIA_30_TICKERS))
    rows = []
    for t in tickers:
        try:
            df = yf.download(
                t, start=start, end=end, auto_adjust=True,
                progress=False, threads=False,
            )
        except Exception:
            continue
        if df.empty or len(df) < 252:
            continue
        # yfinance with auto_adjust returns MultiIndex columns; flatten.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df["ticker"] = t
        rows.append(df[["ticker", "date", "open", "high", "low", "close", "volume"]])
    return pd.concat(rows, ignore_index=True)


def run_djia_phase(
    seed: int,
    method: str,
    output_dir: Path,
    total_timesteps: int = 50_000,
    train_end: str = "2015-12-31",
    test_start: str = "2016-01-04",
    test_end: str = "2020-06-30",
) -> dict:
    """Phase 1: credibility gate on DJIA-30."""
    print(f"[finrl_faithful djia] fetching DJIA-30 prices via yfinance...")
    prices = _fetch_djia_prices(
        start="2009-01-01", end=test_end,
    )
    prices["date"] = pd.to_datetime(prices["date"])
    print(f"[finrl_faithful djia] prices: {prices.shape}, "
          f"tickers={prices['ticker'].nunique()}")
    print(f"[finrl_faithful djia] computing technical indicators...")
    prices = _technical_indicators(prices)
    print(f"[finrl_faithful djia] computing turbulence index...")
    turb = _turbulence_index(prices)
    print(f"[finrl_faithful djia] fetching VIX...")
    vix = _fetch_vix(start="2009-01-01", end=test_end)

    train_df = prices[prices["date"] <= train_end].reset_index(drop=True)
    test_df = prices[
        (prices["date"] >= test_start) & (prices["date"] <= test_end)
    ].reset_index(drop=True)
    train_turb = turb[turb.index <= pd.Timestamp(train_end)]
    test_turb = turb[
        (turb.index >= pd.Timestamp(test_start))
        & (turb.index <= pd.Timestamp(test_end))
    ]
    train_vix = vix[vix.index <= pd.Timestamp(train_end)] if not vix.empty else vix
    test_vix = (
        vix[(vix.index >= pd.Timestamp(test_start))
            & (vix.index <= pd.Timestamp(test_end))]
        if not vix.empty else vix
    )

    tickers = sorted(prices["ticker"].unique().tolist())
    cfg = FinRLEnvConfig()
    train_env = FinRLStockTradingEnv(
        df=train_df, tickers=tickers, cfg=cfg,
        turbulence=train_turb, vix=train_vix,
    )
    test_env = FinRLStockTradingEnv(
        df=test_df, tickers=tickers, cfg=cfg,
        turbulence=test_turb, vix=test_vix,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if method == "ppo":
        agent = train_finrl_ppo(train_env, seed, total_timesteps, device=device)
    elif method == "a2c":
        agent = train_finrl_a2c(train_env, seed, total_timesteps, device=device)
    elif method == "ddpg":
        agent = train_finrl_ddpg(train_env, seed, total_timesteps, device=device)
    else:
        raise ValueError(f"unknown method: {method}")
    perf = evaluate_finrl_env(test_env, agent)
    perf["method"] = method
    perf["seed"] = seed
    print(
        f"[finrl_faithful djia] {method} seed={seed} "
        f"sharpe={perf['sharpe_annualised']:+.3f} "
        f"ann_ret={perf['ann_return']:+.4f} "
        f"ann_vol={perf['ann_vol']:.4f} "
        f"eq={perf['final_equity']:.4f}"
    )
    payload = {
        "phase": "djia",
        "method": method,
        "seed": seed,
        "n_tickers": len(tickers),
        "train_n_days": int(train_df["date"].nunique()),
        "test_n_days": int(test_df["date"].nunique()),
        "perf": perf,
        "config": {
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "total_timesteps": total_timesteps,
            "hmax": cfg.hmax,
            "initial_balance": cfg.initial_balance,
            "transaction_cost_pct": cfg.transaction_cost_pct,
            "reward_scaling": cfg.reward_scaling,
            "turbulence_threshold": cfg.turbulence_threshold,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"djia_{method}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[finrl_faithful djia] wrote {out_path}")
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Faithful FinRL eval.")
    p.add_argument(
        "--phase", type=str, required=True,
        choices=[
            "djia", "universal", "nasdaq100",
            "biotech_nbi", "biotech_nbi_enriched",
        ],
    )
    p.add_argument(
        "--method", type=str, default="ppo",
        choices=["ppo", "a2c", "ddpg"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, default=None, choices=[1, 2, 3, 4, 5])
    p.add_argument(
        "--output-dir-root", type=str,
        default="invar_rl/results/finrl_faithful",
    )
    p.add_argument("--total-timesteps", type=int, default=50_000)
    return p.parse_args()


def _load_universal_panel_prices(
    fold: int,
    universe_k: int = 30,
    panel_kind: str = "lattice_native",
    panel_end: str = "2025-12-31",
    two_regime_val: bool = True,
) -> tuple[pd.DataFrame, list, list, list, list]:
    """Load OHLCV-ish prices for a top-K subset of the lattice_native panel.

    Returns (prices_df, train_dates, val_dates, test_dates, tickers).

    prices_df has columns ticker, date, open, high, low, close, volume.
    Sourced from results/sp500/prices_sp500.parquet (the same raw price
    file used to build the lattice_native panel). The universe is the
    top-``universe_k`` tickers by number of train-segment tradable days
    in the requested fold; fixed across train+val+test of that fold.
    """
    from src.invar import InVARConfig
    from invar_rl.data.lattice_bridge import build_lattice_bridge
    cfg = InVARConfig(fold=fold, seed=42)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    bridge = build_lattice_bridge(cfg)

    # Top-K most-active tickers in the train segment.
    train_active = bridge.tradable[bridge.train_idx].sum(axis=0)
    order = np.argsort(-train_active)
    keep_idx = order[:universe_k]
    tickers = [bridge.tickers[i] for i in keep_idx]

    train_dates = [bridge.dates[d] for d in bridge.train_idx]
    val_dates = [bridge.dates[d] for d in bridge.val_idx]
    test_dates = [bridge.dates[d] for d in bridge.test_idx]
    all_dates = sorted(set(train_dates + val_dates + test_dates))

    # Load raw OHLCV from disk; columns ticker, date, open, high, low,
    # close, adj_close, volume. Panel kind selects the raw price source:
    # lattice_native -> S&P 500 cached prices; nasdaq100 -> NDX prices.
    if panel_kind == "lattice_native":
        prices_path = Path("data/raw/sp500/prices_sp500.parquet")
    elif panel_kind == "nasdaq100":
        prices_path = Path("data/nasdaq100/prices.parquet")
    elif panel_kind == "djia30":
        prices_path = Path("data/djia30/prices.parquet")
    elif panel_kind == "biotech_nbi":
        prices_path = Path("data/biotech_nbi/prices.parquet")
    elif panel_kind == "biotech_nbi_enriched":
        # Same underlying prices + universe as biotech_nbi; only the
        # feature schema differs (22-feature enriched vs 26-feature
        # zero-fill). FinRL only needs OHLCV, so the price source is
        # identical.
        prices_path = Path("data/biotech_nbi/prices.parquet")
    else:
        raise ValueError(f"unsupported panel_kind={panel_kind}")
    raw = pd.read_parquet(prices_path)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["ticker"].isin(tickers)].copy()
    raw_max_date = raw["date"].max()
    all_dates_ts = [pd.Timestamp(d) for d in all_dates]
    # If any requested dates extend past the cached price file (F4 ai-rally
    # 2024, F5 fed-cut 2025-H2), extend via yfinance for the same ticker
    # set so the universal phase covers ALL 5 folds rather than only
    # F1-F3.
    needed_after = [d for d in all_dates_ts if d > raw_max_date]
    if needed_after:
        try:
            import yfinance as yf
            start = (raw_max_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            end_ts = max(needed_after) + pd.Timedelta(days=2)
            end = end_ts.strftime("%Y-%m-%d")
            print(
                f"[finrl_faithful universal] extending prices via "
                f"yfinance for {len(tickers)} tickers, {start}..{end}"
            )
            extra_rows = []
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
                df["ticker"] = t
                df["adj_close"] = df.get("adj close", df["close"])
                extra_rows.append(df[
                    ["ticker", "date", "open", "high", "low", "close",
                     "adj_close", "volume"]
                ])
            if extra_rows:
                extra = pd.concat(extra_rows, ignore_index=True)
                extra["date"] = pd.to_datetime(extra["date"])
                raw = pd.concat([raw, extra], ignore_index=True)
        except ImportError:
            pass
    raw = raw[raw["date"].isin(all_dates_ts)]
    # Coerce missing OHLC: backfill from adj_close if needed.
    if "adj_close" in raw.columns:
        for col in ("open", "high", "low", "close"):
            raw[col] = raw[col].fillna(raw["adj_close"])
    raw = raw[["ticker", "date", "open", "high", "low", "close", "volume"]]
    return raw, train_dates, val_dates, test_dates, tickers


def run_universal_phase(
    fold: int,
    seed: int,
    method: str,
    output_dir: Path,
    total_timesteps: int = 50_000,
    universe_k: int = 30,
) -> dict:
    """Phase 2: stress test on the universal S&P 500 panel under the
    InVAR-RL 5-fold macro-stratified protocol, faithful long-only."""
    print(f"[finrl_faithful universal] fold={fold} loading panel + universe...")
    prices, train_dates, val_dates, test_dates, tickers = (
        _load_universal_panel_prices(fold=fold, universe_k=universe_k)
    )
    print(
        f"[finrl_faithful universal] tickers={len(tickers)} "
        f"train_n={len(train_dates)} val_n={len(val_dates)} "
        f"test_n={len(test_dates)} prices_rows={len(prices)}"
    )
    print(f"[finrl_faithful universal] computing technical indicators...")
    prices = _technical_indicators(prices)
    print(f"[finrl_faithful universal] computing turbulence index...")
    turb = _turbulence_index(prices)
    print(f"[finrl_faithful universal] fetching VIX...")
    all_dates_ts = sorted(set(train_dates) | set(test_dates))
    vix_start = pd.Timestamp(all_dates_ts[0]).strftime("%Y-%m-%d")
    vix_end = (
        pd.Timestamp(all_dates_ts[-1]) + pd.Timedelta(days=2)
    ).strftime("%Y-%m-%d")
    vix = _fetch_vix(start=vix_start, end=vix_end)

    train_df = prices[prices["date"].isin(train_dates)].reset_index(drop=True)
    test_df = prices[prices["date"].isin(test_dates)].reset_index(drop=True)
    train_turb = turb[turb.index.isin(train_dates)]
    test_turb = turb[turb.index.isin(test_dates)]
    train_vix = vix[vix.index.isin(train_dates)] if not vix.empty else vix
    test_vix = vix[vix.index.isin(test_dates)] if not vix.empty else vix

    cfg = FinRLEnvConfig()
    train_env = FinRLStockTradingEnv(
        df=train_df, tickers=tickers, cfg=cfg,
        turbulence=train_turb, vix=train_vix,
    )
    test_env = FinRLStockTradingEnv(
        df=test_df, tickers=tickers, cfg=cfg,
        turbulence=test_turb, vix=test_vix,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if method == "ppo":
        agent = train_finrl_ppo(train_env, seed, total_timesteps, device=device)
    elif method == "a2c":
        agent = train_finrl_a2c(train_env, seed, total_timesteps, device=device)
    elif method == "ddpg":
        agent = train_finrl_ddpg(train_env, seed, total_timesteps, device=device)
    else:
        raise ValueError(f"unknown method: {method}")
    perf = evaluate_finrl_env(test_env, agent)
    perf["method"] = method
    perf["seed"] = seed
    perf["fold"] = fold
    print(
        f"[finrl_faithful universal] fold={fold} {method} seed={seed} "
        f"sharpe={perf['sharpe_annualised']:+.3f} "
        f"ann_ret={perf['ann_return']:+.4f} eq={perf['final_equity']:.4f}"
    )
    payload = {
        "phase": "universal",
        "fold": fold,
        "method": method,
        "seed": seed,
        "n_tickers": len(tickers),
        "n_train_days": len(train_dates),
        "n_test_days": len(test_dates),
        "tickers": tickers,
        "perf": perf,
        "config": {
            "total_timesteps": total_timesteps,
            "universe_k": universe_k,
            "hmax": cfg.hmax,
            "initial_balance": cfg.initial_balance,
            "transaction_cost_pct": cfg.transaction_cost_pct,
            "reward_scaling": cfg.reward_scaling,
            "turbulence_threshold": cfg.turbulence_threshold,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_{method}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[finrl_faithful universal] wrote {out_path}")
    return payload


def run_nasdaq100_phase(
    fold: int,
    seed: int,
    method: str,
    output_dir: Path,
    total_timesteps: int = 50_000,
    universe_k: int = 30,
) -> dict:
    """Phase 5.5 NDX baseline: faithful FinRL on the NASDAQ-100 panel
    under the InVAR-RL 5-fold macro-stratified protocol, long-only top-K.

    Mirrors :func:`run_universal_phase` byte-for-byte; the only change
    is ``panel_kind="nasdaq100"`` passed into
    :func:`_load_universal_panel_prices`, which routes prices to
    ``data/nasdaq100/prices.parquet``. Output is written to
    ``output_dir/fold{F}_{method}_seed{S}.json`` to match the
    rollup script schema.
    """
    print(
        f"[finrl_faithful nasdaq100] fold={fold} loading panel + universe..."
    )
    prices, train_dates, val_dates, test_dates, tickers = (
        _load_universal_panel_prices(
            fold=fold, universe_k=universe_k,
            panel_kind="nasdaq100",
        )
    )
    print(
        f"[finrl_faithful nasdaq100] tickers={len(tickers)} "
        f"train_n={len(train_dates)} val_n={len(val_dates)} "
        f"test_n={len(test_dates)} prices_rows={len(prices)}"
    )
    print(f"[finrl_faithful nasdaq100] computing technical indicators...")
    prices = _technical_indicators(prices)
    print(f"[finrl_faithful nasdaq100] computing turbulence index...")
    turb = _turbulence_index(prices)
    print(f"[finrl_faithful nasdaq100] fetching VIX...")
    all_dates_ts = sorted(set(train_dates) | set(test_dates))
    vix_start = pd.Timestamp(all_dates_ts[0]).strftime("%Y-%m-%d")
    vix_end = (
        pd.Timestamp(all_dates_ts[-1]) + pd.Timedelta(days=2)
    ).strftime("%Y-%m-%d")
    vix = _fetch_vix(start=vix_start, end=vix_end)

    train_df = prices[prices["date"].isin(train_dates)].reset_index(drop=True)
    test_df = prices[prices["date"].isin(test_dates)].reset_index(drop=True)
    train_turb = turb[turb.index.isin(train_dates)]
    test_turb = turb[turb.index.isin(test_dates)]
    train_vix = vix[vix.index.isin(train_dates)] if not vix.empty else vix
    test_vix = vix[vix.index.isin(test_dates)] if not vix.empty else vix

    cfg = FinRLEnvConfig()
    train_env = FinRLStockTradingEnv(
        df=train_df, tickers=tickers, cfg=cfg,
        turbulence=train_turb, vix=train_vix,
    )
    test_env = FinRLStockTradingEnv(
        df=test_df, tickers=tickers, cfg=cfg,
        turbulence=test_turb, vix=test_vix,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if method == "ppo":
        agent = train_finrl_ppo(train_env, seed, total_timesteps, device=device)
    elif method == "a2c":
        agent = train_finrl_a2c(train_env, seed, total_timesteps, device=device)
    elif method == "ddpg":
        agent = train_finrl_ddpg(train_env, seed, total_timesteps, device=device)
    else:
        raise ValueError(f"unknown method: {method}")
    perf = evaluate_finrl_env(test_env, agent)
    perf["method"] = method
    perf["seed"] = seed
    perf["fold"] = fold
    print(
        f"[finrl_faithful nasdaq100] fold={fold} {method} seed={seed} "
        f"sharpe={perf['sharpe_annualised']:+.3f} "
        f"ann_ret={perf['ann_return']:+.4f} eq={perf['final_equity']:.4f}"
    )
    payload = {
        "phase": "nasdaq100",
        "fold": fold,
        "method": method,
        "seed": seed,
        "n_tickers": len(tickers),
        "n_train_days": len(train_dates),
        "n_test_days": len(test_dates),
        "tickers": tickers,
        "perf": perf,
        "config": {
            "total_timesteps": total_timesteps,
            "universe_k": universe_k,
            "panel_kind": "nasdaq100",
            "hmax": cfg.hmax,
            "initial_balance": cfg.initial_balance,
            "transaction_cost_pct": cfg.transaction_cost_pct,
            "reward_scaling": cfg.reward_scaling,
            "turbulence_threshold": cfg.turbulence_threshold,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_{method}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[finrl_faithful nasdaq100] wrote {out_path}")
    return payload


def run_biotech_nbi_enriched_phase(
    fold: int,
    seed: int,
    method: str,
    output_dir: Path,
    total_timesteps: int = 50_000,
    universe_k: int = 30,
) -> dict:
    """Phase 5.5 NBI ENRICHED baseline: faithful FinRL on the biotech NBI
    enriched 22-feature panel under the InVAR-RL 5-fold protocol.

    Byte-identical to :func:`run_biotech_nbi_phase` except the lattice
    bridge is built with ``panel_kind="biotech_nbi_enriched"``. Prices
    are sourced from the same parquet (the enriched panel reuses the
    biotech NBI prices + active mask); only the feature schema differs.
    Output is written to ``output_dir/fold{F}_{method}_seed{S}.json``.
    """
    print(
        f"[finrl_faithful biotech_nbi_enriched] fold={fold} "
        f"loading panel + universe..."
    )
    prices, train_dates, val_dates, test_dates, tickers = (
        _load_universal_panel_prices(
            fold=fold, universe_k=universe_k,
            panel_kind="biotech_nbi_enriched",
        )
    )
    print(
        f"[finrl_faithful biotech_nbi_enriched] tickers={len(tickers)} "
        f"train_n={len(train_dates)} val_n={len(val_dates)} "
        f"test_n={len(test_dates)} prices_rows={len(prices)}"
    )
    print(f"[finrl_faithful biotech_nbi_enriched] computing technical indicators...")
    prices = _technical_indicators(prices)
    print(f"[finrl_faithful biotech_nbi_enriched] computing turbulence index...")
    turb = _turbulence_index(prices)
    print(f"[finrl_faithful biotech_nbi_enriched] fetching VIX...")
    all_dates_ts = sorted(set(train_dates) | set(test_dates))
    vix_start = pd.Timestamp(all_dates_ts[0]).strftime("%Y-%m-%d")
    vix_end = (
        pd.Timestamp(all_dates_ts[-1]) + pd.Timedelta(days=2)
    ).strftime("%Y-%m-%d")
    vix = _fetch_vix(start=vix_start, end=vix_end)

    train_df = prices[prices["date"].isin(train_dates)].reset_index(drop=True)
    test_df = prices[prices["date"].isin(test_dates)].reset_index(drop=True)
    train_turb = turb[turb.index.isin(train_dates)]
    test_turb = turb[turb.index.isin(test_dates)]
    train_vix = vix[vix.index.isin(train_dates)] if not vix.empty else vix
    test_vix = vix[vix.index.isin(test_dates)] if not vix.empty else vix

    cfg = FinRLEnvConfig()
    train_env = FinRLStockTradingEnv(
        df=train_df, tickers=tickers, cfg=cfg,
        turbulence=train_turb, vix=train_vix,
    )
    test_env = FinRLStockTradingEnv(
        df=test_df, tickers=tickers, cfg=cfg,
        turbulence=test_turb, vix=test_vix,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if method == "ppo":
        agent = train_finrl_ppo(train_env, seed, total_timesteps, device=device)
    elif method == "a2c":
        agent = train_finrl_a2c(train_env, seed, total_timesteps, device=device)
    elif method == "ddpg":
        agent = train_finrl_ddpg(train_env, seed, total_timesteps, device=device)
    else:
        raise ValueError(f"unknown method: {method}")
    perf = evaluate_finrl_env(test_env, agent)
    perf["method"] = method
    perf["seed"] = seed
    perf["fold"] = fold
    print(
        f"[finrl_faithful biotech_nbi_enriched] fold={fold} {method} "
        f"seed={seed} sharpe={perf['sharpe_annualised']:+.3f} "
        f"ann_ret={perf['ann_return']:+.4f} eq={perf['final_equity']:.4f}"
    )
    payload = {
        "phase": "biotech_nbi_enriched",
        "fold": fold,
        "method": method,
        "seed": seed,
        "n_tickers": len(tickers),
        "n_train_days": len(train_dates),
        "n_test_days": len(test_dates),
        "tickers": tickers,
        "perf": perf,
        "config": {
            "total_timesteps": total_timesteps,
            "universe_k": universe_k,
            "panel_kind": "biotech_nbi_enriched",
            "hmax": cfg.hmax,
            "initial_balance": cfg.initial_balance,
            "transaction_cost_pct": cfg.transaction_cost_pct,
            "reward_scaling": cfg.reward_scaling,
            "turbulence_threshold": cfg.turbulence_threshold,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_{method}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[finrl_faithful biotech_nbi_enriched] wrote {out_path}")
    return payload


def run_biotech_nbi_phase(
    fold: int,
    seed: int,
    method: str,
    output_dir: Path,
    total_timesteps: int = 50_000,
    universe_k: int = 30,
) -> dict:
    """Phase 5.5 NBI baseline: faithful FinRL on the biotech NBI panel
    under the InVAR-RL 5-fold macro-stratified protocol, long-only top-K.

    Mirrors :func:`run_nasdaq100_phase` byte-for-byte; the only change is
    ``panel_kind="biotech_nbi"`` passed into
    :func:`_load_universal_panel_prices`, which routes prices to
    ``data/biotech_nbi/prices.parquet``. Output is written to
    ``output_dir/fold{F}_{method}_seed{S}.json``.
    """
    print(
        f"[finrl_faithful biotech_nbi] fold={fold} loading panel + universe..."
    )
    prices, train_dates, val_dates, test_dates, tickers = (
        _load_universal_panel_prices(
            fold=fold, universe_k=universe_k,
            panel_kind="biotech_nbi",
        )
    )
    print(
        f"[finrl_faithful biotech_nbi] tickers={len(tickers)} "
        f"train_n={len(train_dates)} val_n={len(val_dates)} "
        f"test_n={len(test_dates)} prices_rows={len(prices)}"
    )
    print(f"[finrl_faithful biotech_nbi] computing technical indicators...")
    prices = _technical_indicators(prices)
    print(f"[finrl_faithful biotech_nbi] computing turbulence index...")
    turb = _turbulence_index(prices)
    print(f"[finrl_faithful biotech_nbi] fetching VIX...")
    all_dates_ts = sorted(set(train_dates) | set(test_dates))
    vix_start = pd.Timestamp(all_dates_ts[0]).strftime("%Y-%m-%d")
    vix_end = (
        pd.Timestamp(all_dates_ts[-1]) + pd.Timedelta(days=2)
    ).strftime("%Y-%m-%d")
    vix = _fetch_vix(start=vix_start, end=vix_end)

    train_df = prices[prices["date"].isin(train_dates)].reset_index(drop=True)
    test_df = prices[prices["date"].isin(test_dates)].reset_index(drop=True)
    train_turb = turb[turb.index.isin(train_dates)]
    test_turb = turb[turb.index.isin(test_dates)]
    train_vix = vix[vix.index.isin(train_dates)] if not vix.empty else vix
    test_vix = vix[vix.index.isin(test_dates)] if not vix.empty else vix

    cfg = FinRLEnvConfig()
    train_env = FinRLStockTradingEnv(
        df=train_df, tickers=tickers, cfg=cfg,
        turbulence=train_turb, vix=train_vix,
    )
    test_env = FinRLStockTradingEnv(
        df=test_df, tickers=tickers, cfg=cfg,
        turbulence=test_turb, vix=test_vix,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if method == "ppo":
        agent = train_finrl_ppo(train_env, seed, total_timesteps, device=device)
    elif method == "a2c":
        agent = train_finrl_a2c(train_env, seed, total_timesteps, device=device)
    elif method == "ddpg":
        agent = train_finrl_ddpg(train_env, seed, total_timesteps, device=device)
    else:
        raise ValueError(f"unknown method: {method}")
    perf = evaluate_finrl_env(test_env, agent)
    perf["method"] = method
    perf["seed"] = seed
    perf["fold"] = fold
    print(
        f"[finrl_faithful biotech_nbi] fold={fold} {method} seed={seed} "
        f"sharpe={perf['sharpe_annualised']:+.3f} "
        f"ann_ret={perf['ann_return']:+.4f} eq={perf['final_equity']:.4f}"
    )
    payload = {
        "phase": "biotech_nbi",
        "fold": fold,
        "method": method,
        "seed": seed,
        "n_tickers": len(tickers),
        "n_train_days": len(train_dates),
        "n_test_days": len(test_dates),
        "tickers": tickers,
        "perf": perf,
        "config": {
            "total_timesteps": total_timesteps,
            "universe_k": universe_k,
            "panel_kind": "biotech_nbi",
            "hmax": cfg.hmax,
            "initial_balance": cfg.initial_balance,
            "transaction_cost_pct": cfg.transaction_cost_pct,
            "reward_scaling": cfg.reward_scaling,
            "turbulence_threshold": cfg.turbulence_threshold,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_{method}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[finrl_faithful biotech_nbi] wrote {out_path}")
    return payload


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir_root) / args.phase
    if args.phase == "djia":
        run_djia_phase(
            seed=args.seed, method=args.method,
            output_dir=out_dir, total_timesteps=args.total_timesteps,
        )
    elif args.phase == "universal":
        if args.fold is None:
            raise SystemExit("--fold is required for universal phase")
        run_universal_phase(
            fold=args.fold, seed=args.seed, method=args.method,
            output_dir=out_dir, total_timesteps=args.total_timesteps,
        )
    elif args.phase == "nasdaq100":
        if args.fold is None:
            raise SystemExit("--fold is required for nasdaq100 phase")
        run_nasdaq100_phase(
            fold=args.fold, seed=args.seed, method=args.method,
            output_dir=out_dir, total_timesteps=args.total_timesteps,
        )
    elif args.phase == "biotech_nbi":
        if args.fold is None:
            raise SystemExit("--fold is required for biotech_nbi phase")
        run_biotech_nbi_phase(
            fold=args.fold, seed=args.seed, method=args.method,
            output_dir=out_dir, total_timesteps=args.total_timesteps,
        )
    elif args.phase == "biotech_nbi_enriched":
        if args.fold is None:
            raise SystemExit(
                "--fold is required for biotech_nbi_enriched phase"
            )
        run_biotech_nbi_enriched_phase(
            fold=args.fold, seed=args.seed, method=args.method,
            output_dir=out_dir, total_timesteps=args.total_timesteps,
        )


if __name__ == "__main__":
    main()
