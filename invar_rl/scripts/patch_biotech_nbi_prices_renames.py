"""Patch known historical rename pairs into ``data/biotech_nbi/prices.parquet``.

Biotech NBI 2014-2025 had effectively zero pure corporate rebrands;
nearly all exits are full acquisitions (Allergan/AbbVie,
Celgene/BMS, Shire/Takeda, etc.) where the acquirer absorbs the
target and yfinance retires the target symbol with no carry-forward.
For acquisition exits there is no modern symbol to back-pull, so the
target ticker remains a yfinance failure and the active_mask gate
naturally drops it on dates after delisting.

The single recoverable rename is:

    MYL -> VTRS   (Mylan + Upjohn => Viatris, 2020-11-16). The modern
                  VTRS ticker on yfinance back-resolves to MYL's full
                  pre-rename OHLCV. The main prices builder already
                  resolves this via ``aliases.parquet``; this script
                  is a safety net for the case where the alias map
                  was incomplete at first run.

The script is generic: any (old, new, change_date) tuple added to
``NEW_ALIASES`` below is patched by pulling the MODERN symbol's full
history and splitting it by the rename date (rows BEFORE go under
the OLD ticker; rows AT/AFTER stay under the new ticker if already
present). Idempotent: a second run skips any old_ticker already in
the prices table.
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
DATA_DIR = REPO_ROOT / "data" / "biotech_nbi"
PRICES_PATH = DATA_DIR / "prices.parquet"
ALIASES_PATH = DATA_DIR / "aliases.parquet"

# old, new, change_date. The MYL -> VTRS entry is documented for
# completeness; if Phase 1's aliases.parquet was honoured by the main
# builder, the row is already present and this script is a no-op.
NEW_ALIASES: List[Tuple[str, str, str]] = [
    ("MYL", "VTRS", "2020-11-16"),
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
