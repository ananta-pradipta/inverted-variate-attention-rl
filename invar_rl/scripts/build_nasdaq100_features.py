"""Build NASDAQ-100 per-ticker per-day technical features for Phase 2.

For each business day from 2014-04-01 (60-day warm-up after 2014-01-01)
to 2025-12-31, persist one row per active ticker with the following
columns (per the Phase 2 spec):

    ticker
    close_return            (1-day log return)
    log_volume_zscore_60d   (60-day rolling z-score of log(volume))
    atr_over_close          (20-day ATR / close)
    rsi_14_over_100         (14-day RSI rescaled to [0,1])
    macd_signal             (MACD signal line, 12/26/9 EMA-based)
    momentum_5d             (5-day cumulative log return)

Plus the 60-day raw OHLCV window keys consumed by the InVAR backbone
are computed but not persisted here (the backbone reads OHLCV from
``data/nasdaq100/prices.parquet`` and slices the 60-day window at
training time). The per-day per-ticker features above are what
downstream daily models consume.

Output:
    data/nasdaq100/features/year={YYYY}/part.parquet
    (year-partitioned parquet to keep file count tractable.)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "nasdaq100"
PRICES_PATH = DATA_DIR / "prices.parquet"
MEMBERSHIP_PATH = DATA_DIR / "membership.parquet"
FEATURES_DIR = DATA_DIR / "features"

WARMUP_START = pd.Timestamp("2014-04-01")
END = pd.Timestamp("2025-12-31")


def _ema(s: pd.Series, span: int) -> pd.Series:
    """Pandas exponential moving average wrapper."""
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Classic Wilder RSI on a single-ticker close series."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha=1/period.
    avg_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_dn = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_dn.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.clip(lower=0.0, upper=100.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
          period: int = 20) -> pd.Series:
    """Wilder ATR over ``period`` days."""
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr


def _features_one(sub: pd.DataFrame) -> pd.DataFrame:
    """Compute per-day features for one ticker (sorted by date asc)."""
    s = sub.sort_values("date").reset_index(drop=True).copy()
    close = s["close"].astype("float64")
    high = s["high"].astype("float64")
    low = s["low"].astype("float64")
    vol = s["volume"].astype("float64").replace(0.0, np.nan)

    s["close_return"] = np.log(close).diff()

    log_vol = np.log(vol)
    log_vol_mean = log_vol.rolling(60, min_periods=30).mean()
    log_vol_std = log_vol.rolling(60, min_periods=30).std().replace(0.0, np.nan)
    s["log_volume_zscore_60d"] = (log_vol - log_vol_mean) / log_vol_std

    atr20 = _atr(high, low, close, period=20)
    s["atr_over_close"] = (atr20 / close.replace(0.0, np.nan))

    s["rsi_14_over_100"] = _rsi(close, period=14) / 100.0

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    s["macd_signal"] = _ema(macd_line, 9)

    s["momentum_5d"] = np.log(close / close.shift(5))

    # 20-day rolling average dollar volume (used by active_mask builder
    # too; included here as a sanity / liquidity feature for reuse).
    dollar_vol = close * vol
    s["adv_dollar_20d"] = dollar_vol.rolling(20, min_periods=10).mean()

    keep = [
        "ticker", "date",
        "close_return", "log_volume_zscore_60d", "atr_over_close",
        "rsi_14_over_100", "macd_signal", "momentum_5d",
        "adv_dollar_20d",
    ]
    return s[keep].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start", default=WARMUP_START.strftime("%Y-%m-%d"),
        help="first business date to persist (rows before are warm-up only)",
    )
    parser.add_argument(
        "--end", default=END.strftime("%Y-%m-%d"),
        help="last business date to persist",
    )
    args = parser.parse_args()

    if not PRICES_PATH.exists():
        print(f"ERROR: missing {PRICES_PATH}; run build_nasdaq100_prices.py "
              f"first.", flush=True)
        return 1

    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    print(f"[features] prices: {len(prices):,} rows; "
          f"{prices['ticker'].nunique()} tickers; "
          f"{prices['date'].min().date()} -> {prices['date'].max().date()}",
          flush=True)

    out_frames: List[pd.DataFrame] = []
    for i, (tk, sub) in enumerate(prices.groupby("ticker", sort=True), 1):
        if len(sub) < 60:
            continue
        feats = _features_one(sub)
        out_frames.append(feats)
        if i % 50 == 0:
            print(f"  [{i}] {tk} features done", flush=True)
    if not out_frames:
        print("[features] no per-ticker frames produced; abort", flush=True)
        return 1

    feats_all = pd.concat(out_frames, ignore_index=True)
    feats_all["date"] = pd.to_datetime(feats_all["date"]).dt.normalize()
    start_ts = pd.to_datetime(args.start)
    end_ts = pd.to_datetime(args.end)
    feats_all = feats_all[
        (feats_all["date"] >= start_ts) & (feats_all["date"] <= end_ts)
    ].copy()
    feats_all["year"] = feats_all["date"].dt.year

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    nan_counts = {}
    for y, g in feats_all.groupby("year"):
        ydir = FEATURES_DIR / f"year={y}"
        ydir.mkdir(parents=True, exist_ok=True)
        g = g.drop(columns=["year"]).sort_values(["date", "ticker"])
        g["ticker"] = g["ticker"].astype("string")
        g.to_parquet(ydir / "part.parquet", index=False)
        # Audit NaN rate (informational; downstream active mask + train
        # gates handle warm-up days properly).
        feat_cols = [
            "close_return", "log_volume_zscore_60d", "atr_over_close",
            "rsi_14_over_100", "macd_signal", "momentum_5d",
        ]
        nan_rate = float(g[feat_cols].isna().mean().mean())
        nan_counts[int(y)] = nan_rate
        print(f"  year={y}: {len(g):,} rows; nan_rate={nan_rate:.4f}",
              flush=True)

    print(f"\n[features] WROTE year-partitioned parquets to {FEATURES_DIR}",
          flush=True)
    print(f"[features] NaN rate per year: {nan_counts}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
