"""Build the NDX (NASDAQ-100) sector cache for the C3 selector.

Produces ``cache/sector_labels/nasdaq100.parquet`` with columns
``ticker`` (str) and ``sector_id`` (int; 0-based; -1 for unknown) using
the canonical 11-sector GICS schema in
``src.models.pretrain_improvements.sector_positives.GICS_SECTOR_ORDER``.

Sources, in priority order (CSV-first; supplements fill missing):

1. Existing ``data/processed/sp500_sector_map.csv`` (high-confidence GICS
   labels). Most NDX names are S&P 500 dual-listed.
2. The ``GICS_SECTOR_SUPPLEMENT`` table inside the C3 module (recent SP500
   additions; includes ABNB / DDOG / CRWD / UBER / ...).
3. A NDX-local supplement defined below (``NDX_SECTOR_SUPPLEMENT``) for
   the historical/foreign tickers that NDX uses but are not in either
   SP500 source. yfinance industry/sector pulled offline on 2026-05-27
   and mapped to GICS top-level sectors with the standard
   yfinance->GICS table; spot-checked against company filings for the
   delisted/renamed entries (CA -> CA Technologies -> Information
   Technology; FB -> Meta -> Communication Services; MYL -> Mylan ->
   Health Care; PCLN -> Booking -> Consumer Discretionary; etc.).

Output schema mirrors the SP500 cache so the existing C3 loader
(``src.models.pretrain_improvements.sector_positives.load_sector_map``)
finds the file under the same key. No source-code edits to the C3
module; this script only writes the parquet.

Usage::

    PYTHONPATH=$PWD python -m invar_rl.scripts.build_nasdaq100_sector_map

Or with a non-default panel parquet::

    PYTHONPATH=$PWD python -m invar_rl.scripts.build_nasdaq100_sector_map \
        --panel-parquet data/nasdaq100/panel_features.parquet \
        --out cache/sector_labels/nasdaq100.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# C3 module is READ-ONLY; we only import the canonical 11-sector order
# and the SP500 supplement table for re-use.
from src.models.pretrain_improvements.sector_positives import (
    GICS_SECTOR_ORDER,
    GICS_SECTOR_SUPPLEMENT,
    UNKNOWN_SECTOR_ID,
)


# ---------------------------------------------------------------------
# NDX-local supplement. Holds tickers that appear in the NDX panel
# but are NOT in data/processed/sp500_sector_map.csv and NOT in the C3
# module's GICS_SECTOR_SUPPLEMENT.
#
# Source: yfinance .info pulled 2026-05-27 (sector + industry), mapped
# to the canonical 11-GICS top-level via the standard yfinance->GICS
# crosswalk:
#   yfinance "Technology" -> GICS "Information Technology"
#   yfinance "Healthcare" -> GICS "Health Care"
#   yfinance "Consumer Cyclical" -> GICS "Consumer Discretionary"
#   yfinance "Consumer Defensive" -> GICS "Consumer Staples"
#   yfinance "Communication Services" -> GICS "Communication Services"
#   yfinance "Financial Services" -> GICS "Financials"
#   yfinance "Industrials" -> GICS "Industrials"
#   yfinance "Basic Materials" -> GICS "Materials"
#   yfinance "Real Estate" -> GICS "Real Estate"
#   yfinance "Utilities" -> GICS "Utilities"
#   yfinance "Energy" -> GICS "Energy"
#
# Delisted/renamed tickers (CA, FB, GMCR, MYL, NLOK, PCLN, SHPG, VIP,
# WLTW) are filled from public knowledge of the index-inclusion-era
# company:
#   CA -> CA Technologies (acquired by Broadcom 2018) -> IT
#   FB -> Facebook / Meta -> Communication Services
#   GMCR -> Green Mountain Coffee Roasters / Keurig -> Consumer Staples
#   MYL -> Mylan (now Viatris) -> Health Care
#   NLOK -> NortonLifeLock (now Gen Digital) -> IT
#   PCLN -> Priceline (now Booking BKNG) -> Consumer Discretionary
#   SHPG -> Shire Pharma (acquired by Takeda 2019) -> Health Care
#   VIP -> VEON / VimpelCom -> Communication Services
#   WLTW -> Willis Towers Watson (now WTW) -> Financials
NDX_SECTOR_SUPPLEMENT: dict[str, str] = {
    "ALNY": "Health Care",
    "ARM": "Information Technology",
    "ASML": "Information Technology",
    "AZN": "Health Care",
    "BATRA": "Communication Services",
    "BATRK": "Communication Services",
    "BIDU": "Communication Services",
    "BMRN": "Health Care",
    "CA": "Information Technology",
    "CCEP": "Consumer Staples",
    "CHKP": "Information Technology",
    "DOCU": "Information Technology",
    "FB": "Communication Services",
    "FER": "Industrials",
    "GFS": "Information Technology",
    "GMCR": "Consumer Staples",
    "INSM": "Health Care",
    "JD": "Consumer Discretionary",
    "LBTYA": "Communication Services",
    "LBTYK": "Communication Services",
    "LCID": "Consumer Discretionary",
    "LILA": "Communication Services",
    "LILAK": "Communication Services",
    "MDB": "Information Technology",
    "MELI": "Consumer Discretionary",
    "MRVL": "Information Technology",
    "MSTR": "Information Technology",
    "MYL": "Health Care",
    "NLOK": "Information Technology",
    "NTES": "Communication Services",
    "OKTA": "Information Technology",
    "PCLN": "Consumer Discretionary",
    "PDD": "Consumer Discretionary",
    "PTON": "Consumer Discretionary",
    "RIVN": "Consumer Discretionary",
    "SHOP": "Information Technology",
    "SHPG": "Health Care",
    "SIRI": "Communication Services",
    "TCOM": "Consumer Discretionary",
    "TEAM": "Information Technology",
    "TRI": "Industrials",
    "VIP": "Communication Services",
    "VOD": "Communication Services",
    "WLTW": "Financials",
    "ZM": "Information Technology",
    "ZS": "Information Technology",
}


def _to_sector_id(sector_str: str) -> int:
    sector_to_id = {s: i for i, s in enumerate(GICS_SECTOR_ORDER)}
    return int(sector_to_id.get(str(sector_str), UNKNOWN_SECTOR_ID))


def build_nasdaq100_sector_map(
    sp500_csv: str = "data/processed/sp500_sector_map.csv",
    panel_parquet: str = "data/nasdaq100/panel_features.parquet",
    out_parquet: str = "cache/sector_labels/nasdaq100.parquet",
) -> pd.DataFrame:
    """Build and persist the NDX sector parquet.

    Returns the cached DataFrame with columns ``ticker`` (str) and
    ``sector_id`` (int; -1 if unknown). Always overwrites the cache.
    """
    panel = pd.read_parquet(panel_parquet)
    ndx_tickers = sorted(panel["ticker"].astype(str).unique().tolist())

    sp500 = pd.read_csv(sp500_csv)
    if not {"ticker", "sector"}.issubset(sp500.columns):
        raise ValueError(
            f"SP500 sector CSV {sp500_csv} missing 'ticker'/'sector' cols;"
            f" got {list(sp500.columns)}"
        )

    # Build a single ticker -> sector_str lookup with priority:
    # SP500 CSV > C3 SP500 supplement > NDX supplement.
    lookup: dict[str, str] = {}
    for tk, sec in zip(
        sp500["ticker"].astype(str), sp500["sector"].astype(str)
    ):
        lookup[tk] = sec
    for tk, sec in GICS_SECTOR_SUPPLEMENT.items():
        if tk not in lookup:
            lookup[tk] = sec
    for tk, sec in NDX_SECTOR_SUPPLEMENT.items():
        if tk not in lookup:
            lookup[tk] = sec

    rows: list[tuple[str, int]] = []
    n_known = 0
    n_unknown = 0
    unknown_tickers: list[str] = []
    for tk in ndx_tickers:
        sec_str = lookup.get(tk)
        if sec_str is None:
            sid = UNKNOWN_SECTOR_ID
            n_unknown += 1
            unknown_tickers.append(tk)
        else:
            sid = _to_sector_id(sec_str)
            if sid == UNKNOWN_SECTOR_ID:
                n_unknown += 1
                unknown_tickers.append(tk)
            else:
                n_known += 1
        rows.append((tk, sid))

    df = pd.DataFrame(rows, columns=["ticker", "sector_id"])
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    df = df.reset_index(drop=True)
    cov = float(n_known) / float(len(ndx_tickers)) if ndx_tickers else 0.0
    print(
        f"[INFO] NDX sector cache: N={len(ndx_tickers)} known={n_known} "
        f"unknown={n_unknown} coverage={cov*100:.2f}%"
    )
    if unknown_tickers:
        preview = ", ".join(unknown_tickers[:10])
        print(f"[INFO] Unknown NDX tickers (first 10): {preview}")
    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[INFO] Wrote {out_path} (rows={len(df)})")
    return df


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build cache/sector_labels/nasdaq100.parquet for C3."
    )
    p.add_argument(
        "--sp500-csv", type=str,
        default="data/processed/sp500_sector_map.csv",
    )
    p.add_argument(
        "--panel-parquet", type=str,
        default="data/nasdaq100/panel_features.parquet",
    )
    p.add_argument(
        "--out", type=str,
        default="cache/sector_labels/nasdaq100.parquet",
    )
    args = p.parse_args()
    df = build_nasdaq100_sector_map(
        sp500_csv=args.sp500_csv,
        panel_parquet=args.panel_parquet,
        out_parquet=args.out,
    )
    # Re-check coverage against the canonical 95% gate.
    n = len(df)
    n_known = int((df["sector_id"] != UNKNOWN_SECTOR_ID).sum())
    cov = float(n_known) / float(n) if n else 0.0
    print(
        f"[INFO] Final coverage: {n_known}/{n} = {cov*100:.2f}% "
        f"(C3 gate >= 95%)"
    )
    if cov < 0.95:
        print(
            "[WARN] Coverage below 95% gate; C3 stage 1 will RAISE. "
            "Add missing tickers to NDX_SECTOR_SUPPLEMENT."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
