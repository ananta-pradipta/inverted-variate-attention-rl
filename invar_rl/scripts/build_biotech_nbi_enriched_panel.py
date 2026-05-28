"""Build the biotech NBI ENRICHED panel using the 22-feature biotech
schema from the RAG-STAR paper.

Mirrors ``src.mtgn.training.panel_enriched.build_enriched_panel`` but
substitutes the NBI universe + NBI prices for the biotech-244 universe
and ``data/raw/prices_universe.parquet``. The output schema is
byte-identical to the biotech 22-col schema used by the canonical RAG-STAR
biotech panel:

    PRICE (9):  log_return, log_return_5d, log_return_20d, log_volume,
                log_volume_ratio_20d, realized_vol_20d, realized_vol_60d,
                high_low_range, close_to_high
    ST (5):     st_volume_24h, st_volume_change_30d, st_bullish_ratio,
                st_sentiment_dispersion, st_labeled_ratio
    FUND (7):   log_market_cap, cash_runway_q, rd_intensity,
                revenue_growth_yoy, cash_to_mc, shares_outstanding_yoy,
                total_assets_growth
    FLAG (1):   has_fundamentals

Total feature_dim = 22.

StockTwits coverage on the NBI universe is ~54% (235 of 433 historical
tickers); fundamentals coverage is ~59% (254 of 433). For tickers
without StockTwits, the ST columns are set to the same neutral defaults
used by the original builder (st_bullish_ratio=0.5, all other ST cols=0).
For tickers without fundamentals, the FUND columns get the per-date
median imputation that the original builder uses, and
``has_fundamentals`` is set to 0.

Output:
    data/biotech_nbi/panel_features_enriched.parquet
        columns: ticker, date, fwd_return_h, plus the 22 FEATURE_COLS.

Forward-return target ``fwd_return_h`` is the 5-day log return.
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
DATA_DIR = REPO_ROOT / "data" / "biotech_nbi"
PRICES_PATH = DATA_DIR / "prices.parquet"
MEMBERSHIP_PATH = DATA_DIR / "membership.parquet"
ST_PATH = REPO_ROOT / "data" / "processed" / "stocktwits_features.parquet"
FUND_PATH = REPO_ROOT / "data" / "raw" / "fundamentals_edgar.parquet"
OUT_PATH = DATA_DIR / "panel_features_enriched.parquet"

HORIZON_DAYS = 5

PRICE_COLS = [
    "log_return",
    "log_return_5d",
    "log_return_20d",
    "log_volume",
    "log_volume_ratio_20d",
    "realized_vol_20d",
    "realized_vol_60d",
    "high_low_range",
    "close_to_high",
]
ST_COLS = [
    "st_volume_24h",
    "st_volume_change_30d",
    "st_bullish_ratio",
    "st_sentiment_dispersion",
    "st_labeled_ratio",
]
FUND_COLS = [
    "log_market_cap",
    "cash_runway_q",
    "rd_intensity",
    "revenue_growth_yoy",
    "cash_to_mc",
    "shares_outstanding_yoy",
    "total_assets_growth",
]
FLAG_COLS = ["has_fundamentals"]
FEATURE_COLS = PRICE_COLS + ST_COLS + FUND_COLS + FLAG_COLS
assert len(FEATURE_COLS) == 22


def _price_features(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Compute the 9 price-derived features + the 5-day forward log return.

    Math copied verbatim from ``src.mtgn.training.panel_enriched._price_features``
    so the schema is byte-for-byte compatible with the biotech-244 panel.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    frames = []
    for _, sub in df.groupby("ticker", sort=False):
        s = sub.copy()
        close = s["close"]
        vol = s["volume"].replace(0, np.nan)
        s["log_return"] = np.log(close).diff()
        s["log_return_5d"] = np.log(close / close.shift(5))
        s["log_return_20d"] = np.log(close / close.shift(20))
        s["log_volume"] = np.log(vol)
        s["log_volume_ratio_20d"] = np.log(
            vol / vol.rolling(20, min_periods=5).mean()
        )
        s["realized_vol_20d"] = s["log_return"].rolling(20, min_periods=5).std()
        s["realized_vol_60d"] = s["log_return"].rolling(60, min_periods=10).std()
        if "high" in s.columns and "low" in s.columns:
            s["high_low_range"] = np.log(s["high"] / s["low"]).clip(upper=0.5)
            s["close_to_high"] = (
                (close - s["low"]) / (s["high"] - s["low"]).replace(0, np.nan)
            )
        else:
            s["high_low_range"] = 0.0
            s["close_to_high"] = 0.5
        s["fwd_return_h"] = np.log(close.shift(-horizon_days) / close)
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def _derive_fundamentals(
    fund: pd.DataFrame, prices: pd.DataFrame,
) -> pd.DataFrame:
    """Compute derived fundamentals from EDGAR quarterly DataFrame.

    Uses ``filed_date`` (public-availability) for forward-fill indexing
    to prevent look-ahead. Historical market cap uses close on the
    filing date. Math copied from
    ``src.mtgn.training.panel_enriched._derive_fundamentals``.
    """
    fund = (
        fund.sort_values(["ticker", "filed_date", "quarter_end"])
        .reset_index(drop=True).copy()
    )
    prices_small = prices[["ticker", "date", "close"]].copy()
    prices_small["date"] = pd.to_datetime(prices_small["date"]).dt.normalize()
    fund["filed_date"] = pd.to_datetime(fund["filed_date"]).dt.normalize()
    fund["quarter_end"] = pd.to_datetime(fund["quarter_end"]).dt.normalize()

    fund = fund.merge(
        prices_small.rename(columns={"date": "filed_date"}),
        how="left", on=["ticker", "filed_date"],
    )

    def _next_trading_close(sub_fund, sub_prices):
        sub_prices = sub_prices.sort_values("date")
        out = []
        for _, r in sub_fund.iterrows():
            if pd.notna(r.get("close")):
                out.append(r["close"])
                continue
            nxt = sub_prices[sub_prices["date"] >= r["filed_date"]]
            out.append(
                nxt.iloc[0]["close"] if len(nxt) > 0 else np.nan
            )
        sub_fund = sub_fund.copy()
        sub_fund["close"] = out
        return sub_fund

    frames = []
    for t, sub in fund.groupby("ticker", sort=False):
        sub_prices = prices_small[prices_small["ticker"] == t]
        sub = _next_trading_close(sub, sub_prices)
        sub["market_cap"] = sub["close"] * sub["shares"]
        sub["date"] = sub["filed_date"]
        sub["log_market_cap"] = np.log(
            sub["market_cap"].replace({0: np.nan})
        )
        sub["burn_q"] = -sub["op_cf"].clip(upper=0)
        sub["cash_runway_q"] = np.where(
            (sub["burn_q"] > 0) & sub["cash"].notna(),
            sub["cash"] / sub["burn_q"], np.nan,
        )
        sub["rd_intensity"] = np.where(
            (sub["market_cap"] > 0) & sub["rd_expense"].notna(),
            sub["rd_expense"] / sub["market_cap"], np.nan,
        )
        sub["revenue_growth_yoy"] = sub["revenue"].pct_change(
            4, fill_method=None
        )
        sub["cash_to_mc"] = np.where(
            (sub["market_cap"] > 0) & sub["cash"].notna(),
            sub["cash"] / sub["market_cap"], np.nan,
        )
        sub["shares_outstanding_yoy"] = sub["shares"].pct_change(
            4, fill_method=None
        )
        sub["total_assets_growth"] = sub["assets"].pct_change(
            4, fill_method=None
        )
        frames.append(sub[["ticker", "date"] + FUND_COLS])
    return pd.concat(frames, ignore_index=True)


def _forward_fill_fundamentals(
    fund: pd.DataFrame, dates,
) -> pd.DataFrame:
    """Per-ticker, forward-fill quarterly values to the daily grid."""
    rows = []
    trading = pd.DatetimeIndex(sorted(dates))
    for t, sub in fund.groupby("ticker"):
        s = sub.sort_values("date").set_index("date")
        s = s[~s.index.duplicated(keep="last")]
        s = s.reindex(trading, method="ffill")
        s["ticker"] = t
        s["date"] = s.index
        rows.append(s.reset_index(drop=True))
    return (
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the biotech NBI enriched panel (22-feature biotech "
            "schema) for InVAR Layer 1 + baseline retraining."
        )
    )
    parser.add_argument(
        "--start", default="2014-09-01",
        help="first business date kept in the panel.",
    )
    parser.add_argument(
        "--end", default="2025-12-31",
        help="last business date kept in the panel.",
    )
    args = parser.parse_args()

    if not PRICES_PATH.exists():
        print(f"ERROR: missing {PRICES_PATH}", flush=True)
        return 1
    if not MEMBERSHIP_PATH.exists():
        print(f"ERROR: missing {MEMBERSHIP_PATH}", flush=True)
        return 1

    # 1. Universe: NBI historical tickers (433 total)
    members = pd.read_parquet(MEMBERSHIP_PATH)
    nbi_tickers = sorted(
        members["ticker"].astype(str).str.upper().unique().tolist()
    )
    print(
        f"[nbi_enriched] NBI universe: {len(nbi_tickers)} tickers",
        flush=True,
    )

    # 2. Prices: NBI prices (subset of universe with yfinance data)
    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    start_ts = pd.to_datetime(args.start)
    end_ts = pd.to_datetime(args.end)
    prices = prices[
        (prices["date"] >= start_ts) & (prices["date"] <= end_ts)
        & prices["ticker"].isin(nbi_tickers)
    ].copy()
    print(
        f"[nbi_enriched] prices: {len(prices):,} rows; "
        f"{prices['ticker'].nunique()} tickers; "
        f"{prices['date'].min().date()} -> {prices['date'].max().date()}",
        flush=True,
    )

    # 3. Compute price-derived features + fwd_return_h
    prices_df = _price_features(prices, HORIZON_DAYS)
    panel_dates = sorted(prices_df["date"].dropna().unique().tolist())

    # 4. StockTwits join (235/433 NBI tickers covered)
    if ST_PATH.exists():
        st = pd.read_parquet(ST_PATH)
        st["ticker"] = st["ticker"].astype(str).str.upper()
        st["date"] = pd.to_datetime(st["date"]).dt.normalize()
        st = st[
            st["ticker"].isin(nbi_tickers)
            & (st["date"] >= start_ts)
            & (st["date"] <= end_ts)
        ]
        st_overlap = st["ticker"].nunique()
        print(
            f"[nbi_enriched] StockTwits: {st_overlap}/{len(nbi_tickers)} "
            f"NBI tickers ({100.0 * st_overlap / len(nbi_tickers):.1f}%)",
            flush=True,
        )
    else:
        print(f"[nbi_enriched] WARN no StockTwits at {ST_PATH}",
              flush=True)
        st = pd.DataFrame(columns=["ticker", "date"] + ST_COLS)
        st_overlap = 0

    panel = prices_df.merge(st, how="left", on=["ticker", "date"])
    panel["st_volume_24h"] = np.log1p(
        panel["st_volume_24h"].fillna(0.0)
    )
    panel["st_volume_change_30d"] = np.log1p(
        panel["st_volume_change_30d"].fillna(1.0).clip(lower=0, upper=1000)
    )
    panel["st_bullish_ratio"] = panel["st_bullish_ratio"].fillna(0.5)
    panel["st_sentiment_dispersion"] = (
        panel["st_sentiment_dispersion"].fillna(0.0)
    )
    panel["st_labeled_ratio"] = panel["st_labeled_ratio"].fillna(0.0)

    # 5. Fundamentals join (254/433 NBI tickers covered)
    if FUND_PATH.exists():
        fund_raw = pd.read_parquet(FUND_PATH)
        fund_raw["ticker"] = fund_raw["ticker"].astype(str).str.upper()
        fund_raw = fund_raw[fund_raw["ticker"].isin(nbi_tickers)]
        fund_overlap = fund_raw["ticker"].nunique()
        print(
            f"[nbi_enriched] Fundamentals: {fund_overlap}/{len(nbi_tickers)} "
            f"NBI tickers ({100.0 * fund_overlap / len(nbi_tickers):.1f}%)",
            flush=True,
        )
        fund = _derive_fundamentals(fund_raw, prices_df)
        fund_daily = _forward_fill_fundamentals(fund, panel_dates)
        if not fund_daily.empty:
            fund_daily = fund_daily[["ticker", "date"] + FUND_COLS]
            panel = panel.merge(
                fund_daily, how="left", on=["ticker", "date"],
            )
        else:
            for c in FUND_COLS:
                panel[c] = np.nan
    else:
        print(f"[nbi_enriched] WARN no fundamentals at {FUND_PATH}",
              flush=True)
        for c in FUND_COLS:
            panel[c] = np.nan
        fund_overlap = 0

    # has_fundamentals flag: 1 if any FUND col is non-null for the row's
    # ticker (this is the original biotech builder's definition: a per-
    # row indicator that the join produced fundamentals, prior to median
    # imputation overwriting NaNs).
    panel["has_fundamentals"] = (
        panel[FUND_COLS].notna().any(axis=1).astype(float)
    )

    # 6. Winsorize fundamental ratios using train-slice percentiles only
    unique_dates = sorted(panel["date"].unique())
    n_train_dates = int(0.65 * len(unique_dates))
    train_cutoff = (
        unique_dates[n_train_dates - 1]
        if n_train_dates > 0 else unique_dates[-1]
    )
    train_mask_row = panel["date"] <= train_cutoff
    ratio_cols = [
        "cash_runway_q", "rd_intensity", "revenue_growth_yoy",
        "cash_to_mc", "shares_outstanding_yoy", "total_assets_growth",
    ]
    for c in ratio_cols:
        if c not in panel.columns:
            continue
        panel[c] = panel[c].replace([np.inf, -np.inf], np.nan)
        vals_train = panel.loc[train_mask_row, c].dropna()
        if len(vals_train) > 100:
            lo = float(vals_train.quantile(0.01))
            hi = float(vals_train.quantile(0.99))
            panel[c] = panel[c].clip(lower=lo, upper=hi)

    # 7. Sector-median impute fundamentals across cross-section per day
    for c in FUND_COLS:
        panel[c] = panel[c].replace([np.inf, -np.inf], np.nan)
        medians = panel.groupby("date")[c].transform("median")
        panel[c] = panel[c].fillna(medians)
    for c in FUND_COLS:
        if panel[c].isna().any():
            train_median = panel.loc[train_mask_row, c].median()
            if pd.isna(train_median):
                train_median = 0.0
            panel[c] = panel[c].fillna(train_median)

    # 8. Drop rows with missing core price features or missing target
    panel = panel.dropna(
        subset=["fwd_return_h", "log_return", "log_volume",
                "realized_vol_20d"],
    )

    # 9. Winsorize returns
    if len(panel) > 0:
        lo = panel["log_return"].quantile(0.005)
        hi = panel["log_return"].quantile(0.995)
        panel["log_return"] = panel["log_return"].clip(lower=lo, upper=hi)
    panel["log_return_5d"] = panel["log_return_5d"].clip(
        lower=-0.8, upper=0.8,
    )
    panel["log_return_20d"] = panel["log_return_20d"].clip(
        lower=-1.5, upper=1.5,
    )

    # 10. Final NaN sweep on feature cols
    for c in FEATURE_COLS:
        panel[c] = (
            pd.to_numeric(panel[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    panel = (
        panel[["ticker", "date", "fwd_return_h"] + FEATURE_COLS]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)
    n_dates = panel["date"].nunique()
    n_tk = panel["ticker"].nunique()
    print(
        f"[nbi_enriched] wrote {OUT_PATH}: {len(panel):,} rows, "
        f"{n_tk} tickers, {n_dates} dates "
        f"(avg active per day {len(panel) / max(1, n_dates):.1f})",
        flush=True,
    )
    print(
        f"[nbi_enriched] feature_dim = {len(FEATURE_COLS)} (22 biotech)",
        flush=True,
    )
    print(
        f"[nbi_enriched] coverage: ST {st_overlap}/{len(nbi_tickers)} "
        f"({100.0 * st_overlap / len(nbi_tickers):.1f}%), "
        f"FUND {fund_overlap}/{len(nbi_tickers)} "
        f"({100.0 * fund_overlap / len(nbi_tickers):.1f}%)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
