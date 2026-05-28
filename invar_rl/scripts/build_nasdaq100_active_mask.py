"""Build the NASDAQ-100 daily active mask for Phase 2.

A (ticker, date) row is ACTIVE iff ALL of:
  - in_index_flag is True per data/nasdaq100/membership.parquet, AND
  - the ticker has a non-NaN close on that date in
    data/nasdaq100/prices.parquet, AND
  - the ticker has >= 60 prior business days of price history (warm-up
    buffer for the 60-day feature window), AND
  - the ticker's 20-day rolling average dollar volume on that date is
    >= 1e6 USD (liquidity filter, see Phase 2 spec; rationale: filters
    out names that are technically in the index for a few days but
    have effectively no tradable depth).

Output: data/nasdaq100/active_mask.parquet
    date     datetime64[ns]
    ticker   string
    active   bool
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "nasdaq100"
PRICES_PATH = DATA_DIR / "prices.parquet"
MEMBERSHIP_PATH = DATA_DIR / "membership.parquet"
OUT_PATH = DATA_DIR / "active_mask.parquet"

MIN_HISTORY_DAYS = 60
MIN_DOLLAR_VOL = 1_000_000.0  # 1e6 USD; 20-day rolling mean


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-history", type=int, default=MIN_HISTORY_DAYS,
    )
    parser.add_argument(
        "--min-dollar-vol", type=float, default=MIN_DOLLAR_VOL,
    )
    args = parser.parse_args()

    if not PRICES_PATH.exists() or not MEMBERSHIP_PATH.exists():
        print("ERROR: prices or membership missing; build them first",
              flush=True)
        return 1

    mem = pd.read_parquet(MEMBERSHIP_PATH)
    mem["date"] = pd.to_datetime(mem["date"]).dt.normalize()
    mem["ticker"] = mem["ticker"].astype(str).str.upper()
    mem = mem[mem["in_index_flag"].astype(bool)].copy()

    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Compute rolling 20-day average dollar volume and a history-count
    # column per ticker.
    frames: List[pd.DataFrame] = []
    for tk, sub in prices.groupby("ticker", sort=False):
        s = sub.sort_values("date").reset_index(drop=True).copy()
        dollar = s["close"].astype("float64") * s["volume"].astype("float64")
        s["adv_dollar_20d"] = dollar.rolling(20, min_periods=10).mean()
        s["history_days"] = (
            s["close"].notna().astype("int64").cumsum() - 1
        )
        frames.append(s[["ticker", "date", "close", "adv_dollar_20d",
                          "history_days"]])
    px = pd.concat(frames, ignore_index=True)

    # Merge with membership (gate: in_index_flag True).
    full = mem.merge(px, how="left", on=["ticker", "date"])

    active = (
        full["close"].notna()
        & (full["history_days"] >= args.min_history)
        & (full["adv_dollar_20d"] >= args.min_dollar_vol)
    )
    out = pd.DataFrame({
        "date": full["date"],
        "ticker": full["ticker"].astype("string"),
        "active": active.astype(bool),
    })
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    active_per_day = out.groupby("date")["active"].sum()
    print(f"[active_mask] WROTE {OUT_PATH}: {len(out):,} rows", flush=True)
    print(f"[active_mask] active per day: "
          f"min={int(active_per_day.min())}, "
          f"median={int(active_per_day.median())}, "
          f"max={int(active_per_day.max())}, "
          f"mean={active_per_day.mean():.2f}", flush=True)
    n_in_index = mem.groupby("date").size()
    drop_per_day = n_in_index - active_per_day.reindex(n_in_index.index,
                                                        fill_value=0)
    print(f"[active_mask] dropped per day (in-index but failed gates): "
          f"min={int(drop_per_day.min())}, "
          f"median={int(drop_per_day.median())}, "
          f"max={int(drop_per_day.max())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
