"""Build the DJIA-30 daily macro-state vector for Phase 2.

The macro vector for the universe-agnostic features (term structure,
credit spread, VIX, ETF returns, etc.) is built using the canonical
S&P 500 universal macro pipeline at
``src.v2.data.universal_macro_features.build_universal_macro_duration_features``
with ``panel_end="2025-12-31"`` so the resulting parquet is BYTE-FOR-
BYTE identical to the S&P 500 universal macro on the overlapping date
range (it is the same builder over the same FRED + sector-ETF inputs).

A single universe-specific column ``cs_avg_pairwise_corr_60d_djia30``
is appended: the cross-sectional average pairwise correlation over a
trailing 60-day window computed from the DJIA-30 active-subset log
returns.

A 5-day release lag is applied to all macro features (each column is
shifted +5 trading days) so that day-t macro inputs only reflect
information available by day t-5. Matches the NASDAQ-100 and S&P 500
macro pipelines byte-for-byte on the overlap.

Inputs:
    data/raw/macro_fred_full.csv          (FRED DGS3MO, DGS2, DGS10, BAA10Y)
    data/raw/sp500/sector_etfs.parquet    (SPY, QQQ, XLK, XLF, ...)
    data/processed/risk_features_sp500.parquet  (vix, vxn, vvix, ...)
    data/djia30/prices.parquet            (Phase 2 prices)
    data/djia30/active_mask.parquet       (Phase 2 active mask)

Outputs:
    data/djia30/macro.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATA_DIR = REPO_ROOT / "data" / "djia30"
SP500_MACRO_PARQUET = REPO_ROOT / "data" / "processed" / "macro_duration_features_sp500.parquet"
OUT_PATH = DATA_DIR / "macro.parquet"

FRED_CSV = REPO_ROOT / "data" / "raw" / "macro_fred_full.csv"
SECTOR_ETFS_PARQUET = REPO_ROOT / "data" / "raw" / "sp500" / "sector_etfs.parquet"
RISK_SP500_PARQUET = REPO_ROOT / "data" / "processed" / "risk_features_sp500.parquet"

REQUIRED_END = pd.Timestamp("2025-12-31")
REQUIRED_START = pd.Timestamp("2014-09-01")
FRED_SERIES = ["DGS3MO", "DGS2", "DGS10", "BAA10Y"]
SECTOR_TICKERS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLY",
                  "XLP", "XLU", "XLRE", "XLC", "XLB", "XLI"]

RELEASE_LAG_DAYS = 5


def _ensure_fred() -> None:
    """Refresh FRED cache to span ``REQUIRED_START`` to ``REQUIRED_END``."""
    need_pull = True
    if FRED_CSV.exists():
        df = pd.read_csv(FRED_CSV, parse_dates=["date"]).set_index("date")
        if (df.index.min() <= REQUIRED_START and df.index.max() >= REQUIRED_END
                and all(s in df.columns for s in FRED_SERIES)):
            need_pull = False
            print(f"[macro] FRED cache OK: {df.index.min().date()} -> "
                  f"{df.index.max().date()}", flush=True)
    if not need_pull:
        return
    from pandas_datareader import data as web
    print(f"[macro] pulling FRED {FRED_SERIES} for "
          f"{REQUIRED_START.date()} -> {REQUIRED_END.date()}", flush=True)
    parts = []
    for s in FRED_SERIES:
        try:
            x = web.DataReader(s, "fred", REQUIRED_START, REQUIRED_END)
            x.columns = [s]
            parts.append(x)
            print(f"  FRED {s}: {len(x)} rows; "
                  f"{x.index.min().date()} -> {x.index.max().date()}",
                  flush=True)
        except Exception as exc:
            print(f"  FRED {s}: error {str(exc)[:120]}", flush=True)
    if not parts:
        raise RuntimeError("FRED fetch failed entirely")
    out = pd.concat(parts, axis=1).sort_index()
    out.index.name = "date"
    FRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FRED_CSV)
    print(f"[macro] wrote {FRED_CSV}: shape={out.shape}", flush=True)


def _ensure_sector_etfs() -> None:
    """Refresh sector ETF parquet to span the required range."""
    need_pull = True
    if SECTOR_ETFS_PARQUET.exists():
        df = pd.read_parquet(SECTOR_ETFS_PARQUET)
        df["date"] = pd.to_datetime(df["date"])
        covered = set(df["ticker"].unique())
        if (df["date"].min() <= REQUIRED_START
                and df["date"].max() >= REQUIRED_END
                and all(t in covered for t in SECTOR_TICKERS)):
            need_pull = False
            print(f"[macro] sector_etfs cache OK: "
                  f"{df['date'].min().date()} -> {df['date'].max().date()}",
                  flush=True)
    if not need_pull:
        return
    import yfinance as yf
    frames = []
    print(f"[macro] pulling sector ETFs {SECTOR_TICKERS}", flush=True)
    for t in SECTOR_TICKERS:
        try:
            d = yf.download(
                t, start=REQUIRED_START.strftime("%Y-%m-%d"),
                end=(REQUIRED_END + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=False, progress=False, threads=False,
            )
            if d is None or d.empty:
                print(f"  {t}: empty", flush=True)
                continue
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = [c[0] for c in d.columns]
            d = d.reset_index()
            d.columns = [str(c).lower() for c in d.columns]
            d = d.rename(columns={"adj close": "adj_close"})
            d["ticker"] = t
            frames.append(d[["ticker", "date", "open", "high", "low",
                              "close", "adj_close", "volume"]])
            print(f"  {t}: {len(d)} rows; "
                  f"{d['date'].min().date()} -> {d['date'].max().date()}",
                  flush=True)
        except Exception as exc:
            print(f"  {t}: error {str(exc)[:120]}", flush=True)
        time.sleep(0.05)
    if not frames:
        raise RuntimeError("sector ETF fetch failed")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    SECTOR_ETFS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SECTOR_ETFS_PARQUET, index=False)
    print(f"[macro] wrote {SECTOR_ETFS_PARQUET}: {len(out):,} rows",
          flush=True)


def _ensure_risk_features() -> None:
    """Extend risk_features_sp500 with FRED VIXCLS through REQUIRED_END."""
    if not RISK_SP500_PARQUET.exists():
        import yfinance as yf
        print("[macro] risk_features_sp500.parquet missing; pulling VIX",
              flush=True)
        d = yf.download(
            "^VIX", start=REQUIRED_START.strftime("%Y-%m-%d"),
            end=(REQUIRED_END + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False, progress=False, threads=False,
        )
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] for c in d.columns]
        d = d.reset_index()
        d.columns = [str(c).lower() for c in d.columns]
        out = pd.DataFrame(index=pd.to_datetime(d["date"]).dt.normalize())
        out["vix"] = d["close"].astype("float64").values
        out["vxn"] = np.nan
        out["vvix"] = np.nan
        out["vix_term_slope"] = np.nan
        out["xbi_rv_20d"] = np.nan
        out["xbi_rv_60d"] = np.nan
        out["vix_5d_change"] = np.nan
        out["xbi_fwd_abs_ret_5d"] = np.nan
        out.index.name = "date"
        RISK_SP500_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(RISK_SP500_PARQUET)
        return

    risk = pd.read_parquet(RISK_SP500_PARQUET)
    risk.index = pd.to_datetime(risk.index)
    if risk.index.max() >= REQUIRED_END:
        print(f"[macro] risk_features_sp500 cache OK to "
              f"{risk.index.max().date()}", flush=True)
        return
    fred = pd.read_csv(FRED_CSV, parse_dates=["date"]).set_index("date")
    if "VIXCLS" in fred.columns:
        new_idx = fred[(fred.index > risk.index.max())
                       & (fred.index <= REQUIRED_END)].index
        if len(new_idx):
            new = pd.DataFrame(index=new_idx, columns=risk.columns,
                                dtype="float64")
            new["vix"] = fred.loc[new_idx, "VIXCLS"].astype("float64").values
            risk = pd.concat([risk, new]).sort_index()
            risk = risk[~risk.index.duplicated(keep="first")]
            risk.to_parquet(RISK_SP500_PARQUET)
            print(f"[macro] extended risk_features_sp500 via FRED VIXCLS: "
                  f"+{len(new)} rows; end={risk.index.max().date()}",
                  flush=True)
            return
    import yfinance as yf
    d = yf.download(
        "^VIX",
        start=(risk.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        end=(REQUIRED_END + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False, progress=False, threads=False,
    )
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d = d.reset_index()
    d.columns = [str(c).lower() for c in d.columns]
    new_idx = pd.to_datetime(d["date"]).dt.normalize()
    new = pd.DataFrame(index=new_idx, columns=risk.columns, dtype="float64")
    new["vix"] = d["close"].astype("float64").values
    risk = pd.concat([risk, new]).sort_index()
    risk = risk[~risk.index.duplicated(keep="first")]
    risk.to_parquet(RISK_SP500_PARQUET)
    print(f"[macro] extended risk_features_sp500 via yfinance: "
          f"+{len(new)} rows; end={risk.index.max().date()}", flush=True)


def _build_universe_corr(
    prices_path: Path, active_mask_path: Path | None,
    window: int = 60,
) -> pd.Series:
    """Compute the universe-specific 60-day avg pairwise correlation.

    For each business day t, take the active DJIA-30 subset (per
    membership and valid close), compute pairwise correlations of the
    trailing ``window``-day log-return windows across active tickers,
    and return the mean off-diagonal Pearson correlation. Causal.
    """
    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices = prices.sort_values(["ticker", "date"])
    prices["log_ret"] = (
        prices.groupby("ticker")["close"].apply(lambda s: np.log(s).diff())
        .reset_index(level=0, drop=True)
    )
    pv = prices.pivot(index="date", columns="ticker", values="log_ret")
    pv = pv.sort_index()

    if active_mask_path is not None and active_mask_path.exists():
        am = pd.read_parquet(active_mask_path)
        am["date"] = pd.to_datetime(am["date"]).dt.normalize()
        am["ticker"] = am["ticker"].astype(str).str.upper()
        gate = am.pivot(
            index="date", columns="ticker", values="active",
        ).reindex(index=pv.index, columns=pv.columns).fillna(False)
        pv_masked = pv.where(gate.astype(bool))
    else:
        pv_masked = pv

    rows: List[tuple] = []
    arr = pv_masked.to_numpy(dtype=np.float64)
    dates = pv_masked.index.tolist()
    for t in range(window - 1, arr.shape[0]):
        win = arr[t - window + 1: t + 1]
        ok = ~np.isnan(win)
        col_active = ok.sum(axis=0) >= max(int(0.6 * window), 20)
        cols = np.nonzero(col_active)[0]
        if cols.size < 5:
            rows.append((dates[t], np.nan))
            continue
        sub = win[:, cols]
        mu = np.nanmean(sub, axis=0)
        sd = np.nanstd(sub, axis=0)
        sd = np.where(sd < 1e-12, 1e-12, sd)
        z = (sub - mu) / sd
        z = np.where(np.isnan(z), 0.0, z)
        n_obs = np.sum(~np.isnan(sub).any(axis=1))
        if n_obs < 10:
            rows.append((dates[t], np.nan))
            continue
        corr = (z.T @ z) / max(z.shape[0] - 1, 1)
        np.fill_diagonal(corr, np.nan)
        rows.append((dates[t], float(np.nanmean(corr))))
    out = pd.Series(
        [r[1] for r in rows],
        index=pd.DatetimeIndex([r[0] for r in rows], name="date"),
        name="cs_avg_pairwise_corr_60d_djia30",
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lag-days", type=int, default=RELEASE_LAG_DAYS,
        help="release-lag in trading days for macro features",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    _ensure_fred()
    _ensure_sector_etfs()
    _ensure_risk_features()

    print("[macro] rebuilding universal macro duration features through "
          f"{REQUIRED_END.date()} ...", flush=True)
    from src.v2.data.universal_macro_features import (
        UniversalMacroDurationConfig, build_universal_macro_duration_features,
    )
    cfg = UniversalMacroDurationConfig()
    cfg.panel_start = "2014-09-01"
    cfg.panel_end = REQUIRED_END.strftime("%Y-%m-%d")
    cfg.fred_cache = FRED_CSV
    cfg.sector_etfs_parquet = SECTOR_ETFS_PARQUET
    cfg.risk_features_parquet = RISK_SP500_PARQUET
    cfg.output_path = SP500_MACRO_PARQUET
    macro = build_universal_macro_duration_features(cfg)

    # ---- Append universe-specific cross-sectional correlation. ----
    prices_path = DATA_DIR / "prices.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(
            f"Phase-2 prices missing: {prices_path}; run "
            f"build_djia30_prices.py first"
        )
    active_mask_path = DATA_DIR / "active_mask.parquet"
    cs_series = _build_universe_corr(
        prices_path, active_mask_path if active_mask_path.exists() else None,
    )

    djia_macro = macro.copy()
    djia_macro["cs_avg_pairwise_corr_60d_djia30"] = cs_series.reindex(
        djia_macro.index,
    )

    if args.lag_days > 0:
        djia_macro = djia_macro.shift(args.lag_days)

    djia_macro = djia_macro.loc[
        djia_macro.index >= pd.Timestamp("2014-01-01")
    ].copy()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    djia_macro.to_parquet(OUT_PATH)
    print(f"[macro] WROTE {OUT_PATH}: shape={djia_macro.shape}; "
          f"{djia_macro.index.min().date()} -> "
          f"{djia_macro.index.max().date()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
