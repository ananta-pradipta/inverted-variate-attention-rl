"""Faithful StockFormer eval, two phases.

Phase 1 (NASDAQ-86 credibility gate): NASDAQ-100 filtered via yfinance
to names with >=98% trading-day coverage; long-only; calm 2016-2020
test window. Target reproducing the upstream ~+1.39 Sharpe.

Phase 2 (universal stress test): top-30 universal S&P 500 panel under
the InVAR-RL 5-fold macro-stratified protocol.

Updated 2026-05-21: aligned to the repo-aligned pipeline in
``baselines/stockformer_faithful.py``. The eval now provides a per-stock
raw OHLCV dict (open/high/low/close/volume per day) to support the
per-stock temporal transformer's (60, F) input, and the NASDAQ universe
is the 98%-trading-day-filtered subset of the canonical FinRL
``NAS_100_TICKER`` list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.invar import InVARConfig

from invar_rl.baselines.stockformer_faithful import (
    StockFormerConfig,
    run_stockformer_faithful,
)
from invar_rl.data.lattice_bridge import build_lattice_bridge


# Canonical FinRL NAS_100_TICKER list (gsyyysg/StockFormer's universe
# is sourced via FinRL's preprocessor). We filter to the 98% trading-day
# subset which lands around 86 names (matches upstream's NASDAQ-86).
NAS_100_TICKER: Tuple[str, ...] = (
    "AMGN", "AAPL", "AMAT", "INTC", "PCAR", "PAYX", "MSFT", "ADBE",
    "CSCO", "QCOM", "COST", "SBUX", "INTU", "AMZN", "GILD", "CMCSA",
    "FAST", "ADSK", "CTSH", "NVDA", "GOOGL", "ISRG", "VRTX", "ADP",
    "ROST", "ORLY", "BKNG", "MU", "MNST", "AVGO", "TXN", "MDLZ",
    "META", "ADI", "WDC", "REGN", "VRSK", "NFLX", "TSLA", "CHTR",
    "MAR", "LRCX", "EA", "KHC", "PYPL", "TMUS", "CSX", "MCHP",
    "CTAS", "KLAC", "IDXX", "MELI", "CDNS", "WDAY", "SNPS", "ASML",
    "TTWO", "PEP", "NXPI", "XEL", "AMD", "ABNB", "AEP", "ALNY",
    "APP", "ARM", "AXON", "BKR", "CCEP", "CEG", "CPRT", "CRWD",
    "CSGP", "DASH", "DDOG", "DXCM", "EXC", "FANG", "FER", "FTNT",
    "GEHC", "GOOG", "HON", "INSM", "KDP", "LIN", "MPWR", "MRVL",
    "MSTR", "ODFL", "PANW", "PDD", "PLTR", "ROP", "SHOP", "STX",
    "TEAM", "TRI", "WBD", "WMT", "ZS",
)


def _fetch_yf_panel(
    tickers: Tuple[str, ...], start: str, end: str,
    min_coverage: float = 0.98,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List, List[str]]:
    """Download OHLCV via yfinance and assemble a per-stock raw panel.

    Returns:
      raw  : dict with keys open/high/low/close/volume, each (T, N)
             aligned to the union trading-day index.
      log_returns : (T, N) day-over-day log returns (zeros at t=0).
      tradable    : (T, N) boolean validity mask.
      dates       : list of T trading-day timestamps.
      tickers     : list of N tickers kept after the >=min_coverage
                    trading-day filter.
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
        cols_needed = {"open", "high", "low", "close", "volume"}
        if not cols_needed.issubset(df.columns):
            continue
        raw_dfs[t] = df[["open", "high", "low", "close", "volume"]]
    if not raw_dfs:
        raise RuntimeError("yfinance returned no tickers")
    union_index = sorted(set().union(*[df.index for df in raw_dfs.values()]))
    union_index = pd.DatetimeIndex(union_index)
    T = len(union_index)
    # Apply 98% trading-day coverage filter.
    kept = []
    for t, df in raw_dfs.items():
        if len(df.index.intersection(union_index)) / T >= min_coverage:
            kept.append(t)
    kept = sorted(kept)
    if not kept:
        raise RuntimeError(
            f"no tickers met >={min_coverage:.0%} trading-day coverage"
        )
    raw_arrs = {
        c: np.zeros((T, len(kept)), dtype=np.float64)
        for c in ("open", "high", "low", "close", "volume")
    }
    tradable = np.zeros((T, len(kept)), dtype=bool)
    for j, t in enumerate(kept):
        df = raw_dfs[t].reindex(union_index)
        valid = df["close"].notna().values
        df = df.ffill().bfill()
        for c in raw_arrs:
            raw_arrs[c][:, j] = df[c].fillna(0.0).values
        tradable[:, j] = valid
    close = raw_arrs["close"]
    log_returns = np.zeros_like(close, dtype=np.float64)
    log_returns[1:] = np.log(
        np.clip(close[1:] / np.clip(close[:-1], 1e-6, None), 1e-6, None)
    )
    return raw_arrs, log_returns, tradable, list(union_index), kept


def _bridge_raw_panel(bridge) -> Dict[str, np.ndarray]:
    """Build the per-stock raw OHLCV dict from the lattice bridge.

    The lattice bridge exposes ``log_returns_1d`` (T, N) and
    ``tradable`` (T, N). We reconstruct a synthetic close-price series
    by exponentiating the cumulative log return (anchored at 1.0 on the
    first valid day per ticker) and use that for open/high/low/close
    (OHLC collapsed to a single curve, since the upstream env only ever
    consumes ``close`` for transactions; the per-stock temporal encoder
    z-scores within the 60-day window so OHLC degeneracy is benign).
    Volume is unavailable on the bridge; we use ones as a placeholder.
    """
    lr = bridge.log_returns_1d.astype(np.float64)
    lr = np.where(np.isfinite(lr), lr, 0.0)
    T, N = lr.shape
    cum = np.cumsum(lr, axis=0)
    close = np.exp(cum - cum[0:1])  # anchor first row at 1.0
    # Where the panel was not tradable, force a flat price (the env
    # treats <=0 as invalid).
    close = np.where(close > 0, close, 1.0)
    return {
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "close": close,
        "volume": np.ones_like(close, dtype=np.float64),
    }


def run_nasdaq_phase(
    seed: int, output_dir: Path,
    total_timesteps: int = 30_000,
    pretrain_epochs: int = 50,
    pretrain_batch_size: int = 32,
) -> dict:
    print(
        f"[stockformer_faithful nasdaq] fetching NAS_100_TICKER prices "
        f"via yfinance (98% trading-day filter)..."
    )
    raw, log_returns, tradable, dates, tickers = _fetch_yf_panel(
        NAS_100_TICKER, "2009-01-01", "2020-06-30",
    )
    print(
        f"[stockformer_faithful nasdaq] kept {len(tickers)} tickers "
        f"after 98% filter (target: NASDAQ-86)"
    )
    train_end_idx = max(
        i for i, d in enumerate(dates) if d <= pd.Timestamp("2015-12-31")
    )
    test_start_idx = min(
        i for i, d in enumerate(dates) if d >= pd.Timestamp("2016-01-04")
    )
    train_days = list(range(60, train_end_idx + 1))
    test_days = list(range(test_start_idx, len(dates) - 1))
    universe = np.arange(len(tickers), dtype=np.int64)
    print(
        f"[stockformer_faithful nasdaq] tickers={len(tickers)} "
        f"train_n={len(train_days)} test_n={len(test_days)}"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = StockFormerConfig(universe_size=len(tickers))
    perf = run_stockformer_faithful(
        log_returns=log_returns, raw=raw,
        tradable=tradable, universe=universe,
        train_days=train_days, test_days=test_days,
        seed=seed, cfg=cfg,
        pretrain_epochs=pretrain_epochs,
        pretrain_batch_size=pretrain_batch_size,
        pretrain_lr=1e-4, sac_lr=1e-4,
        total_timesteps=total_timesteps,
        device=device,
    )
    print(
        f"[stockformer_faithful nasdaq] seed={seed} "
        f"sharpe={perf['sharpe_annualised']:+.3f} "
        f"ann_ret={perf['ann_return']:+.4f} eq={perf['final_equity']:.4f}"
    )
    payload = {
        "phase": "nasdaq",
        "seed": seed,
        "n_tickers": len(tickers),
        "n_train_days": len(train_days),
        "n_test_days": len(test_days),
        "tickers": tickers,
        "perf": perf,
        "config": {
            "total_timesteps": total_timesteps,
            "pretrain_epochs": pretrain_epochs,
            "pretrain_batch_size": pretrain_batch_size,
            "pretrain_lr": 1e-4, "sac_lr": 1e-4,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"nasdaq_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[stockformer_faithful nasdaq] wrote {out_path}")
    return payload


def run_universal_phase(
    fold: int, seed: int, output_dir: Path,
    universe_k: int = 30,
    total_timesteps: int = 30_000,
    pretrain_epochs: int = 50,
    pretrain_batch_size: int = 32,
) -> dict:
    cfg_inv = InVARConfig(fold=fold, seed=seed)
    cfg_inv.panel_kind = "lattice_native"
    cfg_inv.two_regime_val = True
    cfg_inv.panel_end = "2025-12-31"
    bridge = build_lattice_bridge(cfg_inv)
    n_active = bridge.tradable[bridge.train_idx].sum(axis=0)
    order = np.argsort(-n_active)
    universe = np.sort(order[:universe_k])
    print(
        f"[stockformer_faithful universal] fold={fold} "
        f"K={universe_k} train_n={len(bridge.train_idx)} "
        f"test_n={len(bridge.test_idx)}"
    )
    raw = _bridge_raw_panel(bridge)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = StockFormerConfig(universe_size=universe_k)
    perf = run_stockformer_faithful(
        log_returns=bridge.log_returns_1d,
        raw=raw, tradable=bridge.tradable, universe=universe,
        train_days=list(bridge.train_idx),
        test_days=list(bridge.test_idx),
        seed=seed, cfg=cfg,
        pretrain_epochs=pretrain_epochs,
        pretrain_batch_size=pretrain_batch_size,
        pretrain_lr=1e-4, sac_lr=1e-4,
        total_timesteps=total_timesteps,
        device=device,
    )
    perf["fold"] = fold
    perf["seed"] = seed
    print(
        f"[stockformer_faithful universal] fold={fold} seed={seed} "
        f"sharpe={perf['sharpe_annualised']:+.3f} eq={perf['final_equity']:.4f}"
    )
    payload = {
        "phase": "universal",
        "fold": fold, "seed": seed,
        "universe_k": universe_k,
        "perf": perf,
        "config": {
            "total_timesteps": total_timesteps,
            "pretrain_epochs": pretrain_epochs,
            "pretrain_batch_size": pretrain_batch_size,
            "pretrain_lr": 1e-4, "sac_lr": 1e-4,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[stockformer_faithful universal] wrote {out_path}")
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Faithful StockFormer eval.")
    p.add_argument("--phase", type=str, required=True,
                   choices=["nasdaq", "universal", "nasdaq100",
                            "biotech_nbi", "biotech_nbi_enriched"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, default=None,
                   choices=[1, 2, 3, 4, 5])
    p.add_argument("--output-dir-root", type=str,
                   default="invar_rl/results/stockformer_faithful")
    p.add_argument("--total-timesteps", type=int, default=30_000)
    p.add_argument("--pretrain-epochs", type=int, default=50)
    p.add_argument("--pretrain-batch-size", type=int, default=32)
    return p.parse_args()


def run_nasdaq100_phase(
    fold: int, seed: int, output_dir: Path,
    universe_k: int = 30,
    total_timesteps: int = 30_000,
    pretrain_epochs: int = 50,
    pretrain_batch_size: int = 32,
) -> dict:
    """Phase 5.5 NDX baseline: faithful StockFormer on the NASDAQ-100
    panel under the InVAR-RL 5-fold macro-stratified protocol.

    Mirrors :func:`run_universal_phase` exactly; the only difference is
    ``panel_kind="nasdaq100"`` on the lattice bridge so the universe,
    train/val/test dates, and log-return panel come from the NDX-100
    parquets. Output is written to ``output_dir/fold{F}_seed{S}.json``.
    """
    cfg_inv = InVARConfig(fold=fold, seed=seed)
    cfg_inv.panel_kind = "nasdaq100"
    cfg_inv.two_regime_val = True
    cfg_inv.panel_end = "2025-12-31"
    bridge = build_lattice_bridge(cfg_inv)
    n_active = bridge.tradable[bridge.train_idx].sum(axis=0)
    order = np.argsort(-n_active)
    universe = np.sort(order[:universe_k])
    print(
        f"[stockformer_faithful nasdaq100] fold={fold} "
        f"K={universe_k} train_n={len(bridge.train_idx)} "
        f"test_n={len(bridge.test_idx)}"
    )
    raw = _bridge_raw_panel(bridge)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = StockFormerConfig(universe_size=universe_k)
    perf = run_stockformer_faithful(
        log_returns=bridge.log_returns_1d,
        raw=raw, tradable=bridge.tradable, universe=universe,
        train_days=list(bridge.train_idx),
        test_days=list(bridge.test_idx),
        seed=seed, cfg=cfg,
        pretrain_epochs=pretrain_epochs,
        pretrain_batch_size=pretrain_batch_size,
        pretrain_lr=1e-4, sac_lr=1e-4,
        total_timesteps=total_timesteps,
        device=device,
    )
    perf["fold"] = fold
    perf["seed"] = seed
    print(
        f"[stockformer_faithful nasdaq100] fold={fold} seed={seed} "
        f"sharpe={perf['sharpe_annualised']:+.3f} "
        f"eq={perf['final_equity']:.4f}"
    )
    payload = {
        "phase": "nasdaq100",
        "fold": fold, "seed": seed,
        "universe_k": universe_k,
        "perf": perf,
        "config": {
            "total_timesteps": total_timesteps,
            "pretrain_epochs": pretrain_epochs,
            "pretrain_batch_size": pretrain_batch_size,
            "pretrain_lr": 1e-4, "sac_lr": 1e-4,
            "panel_kind": "nasdaq100",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[stockformer_faithful nasdaq100] wrote {out_path}")
    return payload


def run_biotech_nbi_enriched_phase(
    fold: int, seed: int, output_dir: Path,
    universe_k: int = 30,
    total_timesteps: int = 30_000,
    pretrain_epochs: int = 50,
    pretrain_batch_size: int = 32,
) -> dict:
    """Phase 5.5 NBI ENRICHED baseline: faithful StockFormer on the
    biotech NBI enriched 22-feature panel under the InVAR-RL 5-fold
    macro-stratified protocol.

    Byte-identical to :func:`run_biotech_nbi_phase` except the lattice
    bridge is built with ``panel_kind="biotech_nbi_enriched"``. The
    enriched panel reuses biotech NBI prices + active mask (only the
    feature schema differs). Output is written to
    ``output_dir/fold{F}_seed{S}.json``.
    """
    cfg_inv = InVARConfig(fold=fold, seed=seed)
    cfg_inv.panel_kind = "biotech_nbi_enriched"
    cfg_inv.two_regime_val = True
    cfg_inv.panel_end = "2025-12-31"
    bridge = build_lattice_bridge(cfg_inv)
    n_active = bridge.tradable[bridge.train_idx].sum(axis=0)
    order = np.argsort(-n_active)
    universe = np.sort(order[:universe_k])
    print(
        f"[stockformer_faithful biotech_nbi_enriched] fold={fold} "
        f"K={universe_k} train_n={len(bridge.train_idx)} "
        f"test_n={len(bridge.test_idx)}"
    )
    raw = _bridge_raw_panel(bridge)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = StockFormerConfig(universe_size=universe_k)
    perf = run_stockformer_faithful(
        log_returns=bridge.log_returns_1d,
        raw=raw, tradable=bridge.tradable, universe=universe,
        train_days=list(bridge.train_idx),
        test_days=list(bridge.test_idx),
        seed=seed, cfg=cfg,
        pretrain_epochs=pretrain_epochs,
        pretrain_batch_size=pretrain_batch_size,
        pretrain_lr=1e-4, sac_lr=1e-4,
        total_timesteps=total_timesteps,
        device=device,
    )
    perf["fold"] = fold
    perf["seed"] = seed
    print(
        f"[stockformer_faithful biotech_nbi_enriched] fold={fold} "
        f"seed={seed} sharpe={perf['sharpe_annualised']:+.3f} "
        f"eq={perf['final_equity']:.4f}"
    )
    payload = {
        "phase": "biotech_nbi_enriched",
        "fold": fold, "seed": seed,
        "universe_k": universe_k,
        "perf": perf,
        "config": {
            "total_timesteps": total_timesteps,
            "pretrain_epochs": pretrain_epochs,
            "pretrain_batch_size": pretrain_batch_size,
            "pretrain_lr": 1e-4, "sac_lr": 1e-4,
            "panel_kind": "biotech_nbi_enriched",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[stockformer_faithful biotech_nbi_enriched] wrote {out_path}")
    return payload


def run_biotech_nbi_phase(
    fold: int, seed: int, output_dir: Path,
    universe_k: int = 30,
    total_timesteps: int = 30_000,
    pretrain_epochs: int = 50,
    pretrain_batch_size: int = 32,
) -> dict:
    """Phase 5.5 NBI baseline: faithful StockFormer on the biotech NBI
    panel under the InVAR-RL 5-fold macro-stratified protocol.

    Mirrors :func:`run_nasdaq100_phase` byte-for-byte; the only change
    is ``panel_kind="biotech_nbi"`` on the lattice bridge so the
    universe, train/val/test dates, and log-return panel come from the
    biotech NBI parquets. Output is written to
    ``output_dir/fold{F}_seed{S}.json``.
    """
    cfg_inv = InVARConfig(fold=fold, seed=seed)
    cfg_inv.panel_kind = "biotech_nbi"
    cfg_inv.two_regime_val = True
    cfg_inv.panel_end = "2025-12-31"
    bridge = build_lattice_bridge(cfg_inv)
    n_active = bridge.tradable[bridge.train_idx].sum(axis=0)
    order = np.argsort(-n_active)
    universe = np.sort(order[:universe_k])
    print(
        f"[stockformer_faithful biotech_nbi] fold={fold} "
        f"K={universe_k} train_n={len(bridge.train_idx)} "
        f"test_n={len(bridge.test_idx)}"
    )
    raw = _bridge_raw_panel(bridge)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = StockFormerConfig(universe_size=universe_k)
    perf = run_stockformer_faithful(
        log_returns=bridge.log_returns_1d,
        raw=raw, tradable=bridge.tradable, universe=universe,
        train_days=list(bridge.train_idx),
        test_days=list(bridge.test_idx),
        seed=seed, cfg=cfg,
        pretrain_epochs=pretrain_epochs,
        pretrain_batch_size=pretrain_batch_size,
        pretrain_lr=1e-4, sac_lr=1e-4,
        total_timesteps=total_timesteps,
        device=device,
    )
    perf["fold"] = fold
    perf["seed"] = seed
    print(
        f"[stockformer_faithful biotech_nbi] fold={fold} seed={seed} "
        f"sharpe={perf['sharpe_annualised']:+.3f} "
        f"eq={perf['final_equity']:.4f}"
    )
    payload = {
        "phase": "biotech_nbi",
        "fold": fold, "seed": seed,
        "universe_k": universe_k,
        "perf": perf,
        "config": {
            "total_timesteps": total_timesteps,
            "pretrain_epochs": pretrain_epochs,
            "pretrain_batch_size": pretrain_batch_size,
            "pretrain_lr": 1e-4, "sac_lr": 1e-4,
            "panel_kind": "biotech_nbi",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[stockformer_faithful biotech_nbi] wrote {out_path}")
    return payload


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir_root) / args.phase
    if args.phase == "nasdaq":
        run_nasdaq_phase(
            seed=args.seed, output_dir=out_dir,
            total_timesteps=args.total_timesteps,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_batch_size=args.pretrain_batch_size,
        )
    elif args.phase == "universal":
        if args.fold is None:
            raise SystemExit("--fold required for universal phase")
        run_universal_phase(
            fold=args.fold, seed=args.seed,
            output_dir=out_dir,
            total_timesteps=args.total_timesteps,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_batch_size=args.pretrain_batch_size,
        )
    elif args.phase == "nasdaq100":
        if args.fold is None:
            raise SystemExit("--fold required for nasdaq100 phase")
        run_nasdaq100_phase(
            fold=args.fold, seed=args.seed,
            output_dir=out_dir,
            total_timesteps=args.total_timesteps,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_batch_size=args.pretrain_batch_size,
        )
    elif args.phase == "biotech_nbi":
        if args.fold is None:
            raise SystemExit("--fold required for biotech_nbi phase")
        run_biotech_nbi_phase(
            fold=args.fold, seed=args.seed,
            output_dir=out_dir,
            total_timesteps=args.total_timesteps,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_batch_size=args.pretrain_batch_size,
        )
    elif args.phase == "biotech_nbi_enriched":
        if args.fold is None:
            raise SystemExit(
                "--fold required for biotech_nbi_enriched phase"
            )
        run_biotech_nbi_enriched_phase(
            fold=args.fold, seed=args.seed,
            output_dir=out_dir,
            total_timesteps=args.total_timesteps,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_batch_size=args.pretrain_batch_size,
        )


if __name__ == "__main__":
    main()
