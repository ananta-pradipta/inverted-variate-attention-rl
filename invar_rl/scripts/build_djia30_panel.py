"""Build the DJIA-30 26-col canonical panel for InVAR Layer 1.

Mirrors the lattice_native 26-col schema (see
``src.v2.data.lattice_native_panel.FEATURE_COLS``) so the canonical
InVAR model code can ingest the DJIA-30 panel with no architecture
changes (same ``feature_dim = 26``).

Computes the 10 PRICE_VOL columns from ``data/djia30/prices.parquet``
verbatim from ``src.lattice.data.build_panel._compute_price_vol``
math. Zero-fills the remaining 16 columns (4 distress, 4 intangible,
3 other fundamental, 3 catalyst, 2 flag) because the DJIA-30 universe
does not ship a fundamentals / catalyst feed at this stage. The flag
columns ``has_fundamentals`` and ``has_stocktwits`` are set to 0 to
mark these slots as universe-missing.

Output:
    data/djia30/panel_features.parquet
        columns: ticker, date, fwd_return_h, plus the 26 FEATURE_COLS.

Forward-return target ``fwd_return_h`` is the 5-day log return
``log(close[t+5] / close[t])`` (panel ``horizon_days = 5``).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "djia30"
PRICES_PATH = DATA_DIR / "prices.parquet"
ACTIVE_MASK_PATH = DATA_DIR / "active_mask.parquet"
OUT_PATH = DATA_DIR / "panel_features.parquet"

HORIZON_DAYS = 5

FEATURE_COLS = [
    # PRICE_VOL (10)
    "log_return", "log_return_5d", "log_return_20d",
    "log_volume", "log_volume_ratio_20d",
    "realized_vol_20d", "realized_vol_60d",
    "high_low_range", "close_to_high_5d", "amihud_illiquidity_20d",
    # DISTRESS (4)
    "interest_coverage", "net_debt_to_ebitda",
    "fcf_yield", "current_ratio",
    # INTANG (4)
    "rd_to_sales", "sga_to_sales", "gross_profitability", "capex_to_sales",
    # OTHER FUND (3)
    "log_market_cap", "book_to_market", "asset_growth_yoy",
    # CATALYST (3)
    "days_to_next_catalyst_sin", "days_to_next_catalyst_cos",
    "catalyst_type_id",
    # FLAGS (2)
    "has_fundamentals", "has_stocktwits",
]
assert len(FEATURE_COLS) == 26

ZERO_FILL_COLS = FEATURE_COLS[10:]


def _compute_price_vol_one(sub: pd.DataFrame) -> pd.DataFrame:
    """Compute the 10 PRICE_VOL columns for one ticker (sorted asc).

    Math copied verbatim from
    ``src.lattice.data.build_panel._compute_price_vol`` so cross-universe
    parity is byte-for-byte where the underlying OHLCV agrees.

    Args:
        sub: one ticker's OHLCV rows (ticker, date, open, high, low,
            close, volume) sorted by date.

    Returns:
        DataFrame with ticker, date, the 10 PRICE_VOL columns, and
        fwd_return_h (5-day forward log return). NaN where the rolling
        window has too few observations.
    """
    s = sub.sort_values("date").reset_index(drop=True).copy()
    close = s["close"].astype("float64")
    hi = s["high"].astype("float64")
    lo = s["low"].astype("float64")
    vol = s["volume"].astype("float64").replace(0.0, np.nan)
    dollar_vol = (close * vol).replace(0.0, np.nan)

    s["log_return"] = np.log(close).diff()
    s["log_return_5d"] = np.log(close / close.shift(5))
    s["log_return_20d"] = np.log(close / close.shift(20))
    s["log_volume"] = np.log(vol)
    s["log_volume_ratio_20d"] = np.log(
        vol / vol.rolling(20, min_periods=5).mean()
    )
    s["realized_vol_20d"] = s["log_return"].rolling(20, min_periods=5).std()
    s["realized_vol_60d"] = s["log_return"].rolling(60, min_periods=10).std()
    s["high_low_range"] = np.log(hi / lo).clip(upper=0.5)
    close_to_high_5d_max = hi.rolling(5, min_periods=2).max()
    s["close_to_high_5d"] = (close / close_to_high_5d_max).fillna(1.0)
    ret_abs = s["log_return"].abs()
    s["amihud_illiquidity_20d"] = (
        (ret_abs / dollar_vol.replace(0.0, np.nan))
        .rolling(20, min_periods=5).mean()
    ).clip(upper=1e-3)

    s["fwd_return_h"] = np.log(close.shift(-HORIZON_DAYS) / close)

    keep = ["ticker", "date", "fwd_return_h"] + FEATURE_COLS[:10]
    return s[keep].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start", default="2014-09-01",
        help="first business date kept in the panel (warm-up before is dropped).",
    )
    parser.add_argument(
        "--end", default="2025-12-31",
        help="last business date kept in the panel.",
    )
    parser.add_argument(
        "--restrict-to-active", action="store_true",
        help="If set, drop rows where active_mask is False. Default keeps "
             "all rows so the trainer's tradable_mask is the single source "
             "of truth (mirrors the lattice_native panel which keeps all "
             "rows and lets the trainer filter).",
    )
    args = parser.parse_args()

    if not PRICES_PATH.exists():
        print(f"ERROR: missing {PRICES_PATH}", flush=True)
        return 1

    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    print(
        f"[djia30_panel] prices: {len(prices):,} rows; "
        f"{prices['ticker'].nunique()} tickers; "
        f"{prices['date'].min().date()} -> {prices['date'].max().date()}",
        flush=True,
    )

    out_frames = []
    for i, (tk, sub) in enumerate(prices.groupby("ticker", sort=True), 1):
        if len(sub) < 60:
            continue
        out_frames.append(_compute_price_vol_one(sub))
        if i % 10 == 0:
            print(f"  [{i}] {tk} done", flush=True)
    panel = pd.concat(out_frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    start_ts = pd.to_datetime(args.start)
    end_ts = pd.to_datetime(args.end)
    panel = panel[(panel["date"] >= start_ts) & (panel["date"] <= end_ts)].copy()

    pre_drop = len(panel)
    panel = panel.dropna(subset=["fwd_return_h"]).reset_index(drop=True)
    print(
        f"[djia30_panel] dropped {pre_drop - len(panel):,} rows with "
        f"NaN fwd_return_h; remaining: {len(panel):,}",
        flush=True,
    )

    for c in FEATURE_COLS[:10]:
        panel[c] = (
            pd.to_numeric(panel[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    for c in ZERO_FILL_COLS:
        panel[c] = 0.0

    panel = panel[["ticker", "date", "fwd_return_h"] + FEATURE_COLS].copy()
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)
    print(
        f"[djia30_panel] wrote {OUT_PATH}: "
        f"{len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
        f"{panel['date'].nunique()} dates",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
