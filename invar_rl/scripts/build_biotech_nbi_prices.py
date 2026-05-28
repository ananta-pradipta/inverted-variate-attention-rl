"""Build the biotech NBI daily OHLCV price panel for Phase 2.

Reads :mod:`data/biotech_nbi/membership.parquet` for the historical
ticker universe (433 unique symbols across 2014-2025) and downloads
daily bars from yfinance with ``auto_adjust=True`` so the close column
is already split- and dividend-adjusted.

Output: ``data/biotech_nbi/prices.parquet`` (long-format).

Schema (matches ``data/raw/sp500/prices_sp500.parquet``):
    ticker        (string)
    date          (datetime64[ns])
    open, high, low, close, adj_close   (float64)
    volume        (float64)

The MYL -> VTRS rename (2020-11-16; Mylan + Upjohn => Viatris) is
resolved via ``aliases.parquet``. The modern symbol VTRS carries
pre-rename history transparently and is split by the rename date at
persist time.

Biotech NBI is a high-turnover universe (~25-35% expected yfinance
failure rate vs ~14% on NASDAQ-100 vs ~2% on DJIA-30) because
many delisted small-cap biotech names lose their yfinance presence
after acquisition. The :mod:`patch_biotech_nbi_prices_renames`
script is run AFTER this builder to recover any aliases that
yfinance no longer resolves to a modern symbol.

This script is idempotent; it writes to a temp parquet then
atomically renames. Use ``--no-cache`` to force a full re-download.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "biotech_nbi"
MEMBERSHIP_PATH = DATA_DIR / "membership.parquet"
ALIASES_PATH = DATA_DIR / "aliases.parquet"
OUT_PATH = DATA_DIR / "prices.parquet"
START = "2014-01-01"
END = "2025-12-31"


def _modern_symbol(t: str, aliases: dict) -> str:
    """Return the modern (post-rename) symbol for a historical ticker."""
    return aliases.get(t, t)


def _fetch(symbol: str, start: str, end: str,
           max_retries: int = 3) -> pd.DataFrame | None:
    """Download OHLCV for ``symbol`` from yfinance with retries.

    Uses ``auto_adjust=True`` so ``close`` is already split- and
    dividend-adjusted. ``adj_close`` is set equal to ``close`` (yfinance
    drops the separate column when auto_adjust=True; we keep both for
    schema parity with prices_sp500.parquet).
    """
    import yfinance as yf
    last_err = None
    for attempt in range(max_retries):
        try:
            df = yf.download(
                symbol, start=start, end=end,
                auto_adjust=True, progress=False, threads=False,
            )
            if df is None or df.empty:
                last_err = "empty frame"
                time.sleep(0.5 * (attempt + 1))
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]
            if "adj close" in df.columns:
                df = df.rename(columns={"adj close": "adj_close"})
            if "adj_close" not in df.columns:
                df["adj_close"] = df["close"]
            df["ticker"] = symbol
            keep = ["ticker", "date", "open", "high", "low",
                    "close", "adj_close", "volume"]
            df = df[keep]
            df["date"] = pd.to_datetime(df["date"])
            for c in ("open", "high", "low", "close", "adj_close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
            return df
        except Exception as exc:
            last_err = str(exc)[:120]
            time.sleep(0.7 * (attempt + 1))
    print(f"  [fail] {symbol}: {last_err}", flush=True)
    return None


def _load_universe() -> Tuple[List[str], dict]:
    """Return (unique tickers in membership, alias old->new map)."""
    mem = pd.read_parquet(MEMBERSHIP_PATH)
    tickers = sorted(set(mem["ticker"].astype(str).str.upper().tolist()))
    aliases: dict = {}
    if ALIASES_PATH.exists():
        al = pd.read_parquet(ALIASES_PATH)
        for _, r in al.iterrows():
            aliases[str(r["old_ticker"]).upper()] = str(r["new_ticker"]).upper()
    return tickers, aliases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start", default=START, help="start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end", default=END, help="end date YYYY-MM-DD",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="ignore existing prices.parquet and re-download everything",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tickers, aliases = _load_universe()
    print(f"[prices] {len(tickers)} unique tickers in membership", flush=True)
    print(f"[prices] aliases: {aliases}", flush=True)

    # Collapse old/new pairs to a single modern symbol so yfinance is
    # only hit once per economic entity. Expand back to as-of-date
    # symbols at persist time so the rows align with membership.parquet.
    modern_symbols: List[str] = []
    for t in tickers:
        m = _modern_symbol(t, aliases)
        if m not in modern_symbols:
            modern_symbols.append(m)
    print(f"[prices] {len(modern_symbols)} modern symbols to download",
          flush=True)

    existing: pd.DataFrame | None = None
    if OUT_PATH.exists() and not args.no_cache:
        existing = pd.read_parquet(OUT_PATH)
        existing["date"] = pd.to_datetime(existing["date"])
        max_end = pd.Timestamp(args.end)
        cov = (
            existing.groupby("ticker")["date"].agg(["min", "max"])
        )
        already_full = set(
            cov[(cov["min"] <= pd.Timestamp(args.start) + pd.Timedelta(days=30))
                & (cov["max"] >= max_end - pd.Timedelta(days=7))].index
        )
        print(f"[prices] cache present: {len(existing):,} rows; "
              f"{len(already_full)} symbols already cover the window",
              flush=True)
    else:
        already_full = set()

    frames: List[pd.DataFrame] = []
    if existing is not None and not args.no_cache:
        keep_cache = existing[existing["ticker"].isin(already_full)].copy()
        frames.append(keep_cache)

    failures: List[str] = []
    n_pull = 0
    to_pull = [s for s in modern_symbols if s not in already_full]
    for i, sym in enumerate(to_pull, 1):
        df = _fetch(sym, args.start, args.end)
        if df is None or df.empty:
            failures.append(sym)
            continue
        frames.append(df)
        n_pull += 1
        if i % 25 == 0 or i == len(to_pull):
            print(f"  [{i}/{len(to_pull)}] pulled OK={n_pull} "
                  f"fail={len(failures)}", flush=True)
        time.sleep(0.05)

    merged = pd.concat(frames, ignore_index=True)
    merged["ticker"] = merged["ticker"].astype(str).str.upper()
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = (
        merged.drop_duplicates(subset=["ticker", "date"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    # Expand modern symbols back to as-of-date symbols for any alias
    # pair: rows BEFORE change_date are persisted under the OLD ticker
    # so they align with membership.parquet (which uses as-of-date
    # symbols).
    if ALIASES_PATH.exists():
        al = pd.read_parquet(ALIASES_PATH)
        if not al.empty:
            extra = []
            for _, r in al.iterrows():
                old_t = str(r["old_ticker"]).upper()
                new_t = str(r["new_ticker"]).upper()
                chg = pd.Timestamp(r["change_date"])
                pre = merged[(merged["ticker"] == new_t)
                              & (merged["date"] < chg)].copy()
                if pre.empty:
                    continue
                pre["ticker"] = old_t
                extra.append(pre)
                drop_mask = (
                    (merged["ticker"] == new_t)
                    & (merged["date"] < chg)
                )
                merged = merged[~drop_mask].copy()
            if extra:
                merged = pd.concat([merged] + extra, ignore_index=True)
                merged = (
                    merged.sort_values(["ticker", "date"])
                    .reset_index(drop=True)
                )

    merged["ticker"] = merged["ticker"].astype("string")
    merged["date"] = pd.to_datetime(merged["date"])
    for c in ("open", "high", "low", "close", "adj_close", "volume"):
        merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float64")

    tmp = OUT_PATH.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, OUT_PATH)
    print(f"\n[prices] WROTE {OUT_PATH}: {len(merged):,} rows, "
          f"{merged['ticker'].nunique()} tickers, "
          f"range {merged['date'].min().date()} -> "
          f"{merged['date'].max().date()}", flush=True)
    print(f"[prices] failures ({len(failures)}): {failures}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
