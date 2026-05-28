"""Patch known historical rename pairs into ``data/djia30/prices.parquet``.

The first prices pull misses two tickers because yfinance does not
back-resolve the pre-rename symbol any longer:

    DWDP -> DD   (the DowDuPont merger entity, 2017-09-01 to 2019-04-01,
                  whose history is carried forward by the modern DD
                  ticker; pre-2019-04-02 rows of DD ARE the DowDuPont
                  history under the merged ticker, so we copy them
                  under the DWDP symbol for membership alignment).
    WBA  -> None (Walgreens Boots Alliance was taken private 2024-12; the
                  symbol is no longer served by yfinance or stooq-free.
                  Documented in reports/djia30/phase_2_report.md; the
                  active-mask gate will exclude WBA at evaluation time
                  with no recovery available without a paid vendor.)

For each recoverable pair we pull the MODERN symbol's full history
from yfinance and split it by the rename date: rows BEFORE the rename
go under the OLD ticker, rows ON OR AFTER go under the new ticker.

This is a thin recovery patch on top of the main prices builder. It
also updates ``data/djia30/aliases.parquet`` to record the renames so
downstream membership joins work transparently.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "djia30"
PRICES_PATH = DATA_DIR / "prices.parquet"
ALIASES_PATH = DATA_DIR / "aliases.parquet"

# old, new, change_date. DowDuPont -> post-split DD on 2019-04-02 (the
# day after the spin-off effective date 2019-04-01).
NEW_ALIASES: List[Tuple[str, str, str]] = [
    ("DWDP", "DD", "2019-04-02"),
]


def _fetch(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Download with auto_adjust=True; same shape as the main builder."""
    import yfinance as yf
    for attempt in range(3):
        try:
            df = yf.download(
                symbol, start=start, end=end,
                auto_adjust=True, progress=False, threads=False,
            )
            if df is None or df.empty:
                time.sleep(0.7 * (attempt + 1))
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
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            for c in ("open", "high", "low", "close", "adj_close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
            return df
        except Exception as exc:
            print(f"  fetch error {symbol} attempt {attempt}: "
                  f"{str(exc)[:120]}", flush=True)
            time.sleep(0.5 * (attempt + 1))
    return None


def main() -> int:
    if not PRICES_PATH.exists():
        print(f"ERROR: missing {PRICES_PATH}", flush=True)
        return 1
    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()

    aliases = (
        pd.read_parquet(ALIASES_PATH)
        if ALIASES_PATH.exists()
        else pd.DataFrame(
            columns=["old_ticker", "new_ticker", "change_date"]
        )
    )
    if not aliases.empty:
        aliases["change_date"] = pd.to_datetime(aliases["change_date"])

    new_rows = []
    new_alias_rows = []
    for old_t, new_t, chg in NEW_ALIASES:
        chg_ts = pd.Timestamp(chg)
        have_old = (prices["ticker"] == old_t).any()
        if have_old:
            print(f"[patch] {old_t} already present; skip", flush=True)
            continue
        df = _fetch(new_t, "2014-01-01", "2025-12-31")
        if df is None or df.empty:
            print(f"[patch] {new_t} fetch failed; skip", flush=True)
            continue
        # Pre-rename rows are persisted under the OLD ticker. The DD
        # rows AT/AFTER the rename remain under DD where they already
        # exist (the main builder pulled DD separately as a current
        # DJIA member).
        df_pre = df[df["date"] < chg_ts].copy()
        df_pre["ticker"] = old_t
        new_rows.append(df_pre)
        new_alias_rows.append((old_t, new_t, chg_ts))
        print(f"[patch] {old_t}/{new_t}: pre={len(df_pre)} rows persisted "
              f"under {old_t}", flush=True)

    if not new_rows:
        print("[patch] nothing to patch", flush=True)
        return 0

    add = pd.concat(new_rows, ignore_index=True)
    merged = pd.concat([prices, add], ignore_index=True)
    merged["ticker"] = merged["ticker"].astype(str).str.upper()
    merged = (
        merged.drop_duplicates(subset=["ticker", "date"], keep="first")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    merged["ticker"] = merged["ticker"].astype("string")
    merged.to_parquet(PRICES_PATH, index=False)
    print(f"[patch] WROTE {PRICES_PATH}: {len(merged):,} rows; "
          f"{merged['ticker'].nunique()} tickers", flush=True)

    if new_alias_rows:
        new_al = pd.DataFrame(
            new_alias_rows,
            columns=["old_ticker", "new_ticker", "change_date"],
        )
        new_al["change_date"] = pd.to_datetime(new_al["change_date"])
        all_al = pd.concat([aliases, new_al], ignore_index=True)
        all_al = all_al.drop_duplicates(
            subset=["old_ticker"], keep="last",
        ).reset_index(drop=True)
        all_al.to_parquet(ALIASES_PATH, index=False)
        print(f"[patch] aliases updated: {len(all_al)} rows", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
