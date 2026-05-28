"""Build the 550 x 550 GICS sector adjacency matrix for DeepTrader.

DeepTrader's Asset Scoring Unit (Wang et al., AAAI 2021) requires an
``industry_classification.npy`` adjacency over the asset universe. For
our lattice_native S&P 500 panel (550 tickers, plan calls it 600), we
source the GICS sector from ``data/lattice/processed/cohorts.parquet``
which already has a ``sector`` column for every ticker; this avoids a
yfinance fetch entirely (deterministic, no rate limits, complete
coverage).

Adjacency rule: ``A[i, j] = 1`` if tickers ``i`` and ``j`` share the
same GICS sector or ``i == j``; ``A[i, j] = 0`` otherwise. The diagonal
is unit; the matrix is symmetric. The adaptive-adjacency mechanism in
the GCN then refines this prior at training time.

Fallback rule (only triggers if cohorts.parquet has >5 percent null
sectors, which it currently does not for the lattice_native universe):
build the adjacency from 60-day rolling close-to-close correlation
thresholded at 0.5, using ``data/lattice/processed/panel_features.parquet``
log-returns.

Outputs:
- ``data/processed/sp500_sector_adjacency.npy`` (np.float32, NxN)
- ``data/processed/sp500_sector_map.csv`` (ticker, sector)

Usage::

    python3 invar_rl/scripts/build_sector_adjacency.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
COHORTS_PATH = REPO_ROOT / "data" / "lattice" / "processed" / "cohorts.parquet"
PANEL_PATH = REPO_ROOT / "data" / "lattice" / "processed" / "panel_features.parquet"
OUT_DIR = REPO_ROOT / "data" / "processed"
ADJACENCY_OUT = OUT_DIR / "sp500_sector_adjacency.npy"
SECTOR_MAP_OUT = OUT_DIR / "sp500_sector_map.csv"

NULL_TOLERANCE = 0.05
CORR_THRESHOLD = 0.5
CORR_WINDOW_DAYS = 60


def _load_sector_map() -> pd.DataFrame:
    """Load the last-observed sector per ticker from cohorts.parquet.

    Returns:
        DataFrame with columns (ticker, sector), one row per ticker,
        sorted alphabetically by ticker.
    """
    df = pd.read_parquet(COHORTS_PATH, columns=["ticker", "date", "sector"])
    df = df.sort_values(["ticker", "date"])
    last = df.groupby("ticker", as_index=False).last()[["ticker", "sector"]]
    last = last.sort_values("ticker").reset_index(drop=True)
    return last


def _fallback_correlation_adjacency(tickers: list[str]) -> np.ndarray:
    """Build a 60-day rolling-correlation thresholded adjacency.

    Only invoked when GICS lookup fails for more than ``NULL_TOLERANCE``
    of tickers. Reads ``log_return`` from the lattice panel, pivots to
    a (T, N) returns matrix, and thresholds the full-sample Pearson
    correlation at ``CORR_THRESHOLD``.

    Args:
        tickers: Sorted list of ticker symbols (length N).

    Returns:
        Symmetric (N, N) np.float32 adjacency with unit diagonal.
    """
    panel = pd.read_parquet(PANEL_PATH, columns=["ticker", "date", "log_return"])
    panel = panel[panel["ticker"].isin(tickers)]
    mat = panel.pivot(index="date", columns="ticker", values="log_return")
    mat = mat.reindex(columns=tickers)
    # Use the most recent CORR_WINDOW_DAYS rows, fill NaN with 0 so the
    # correlation is computed on a clean matrix.
    window = mat.tail(CORR_WINDOW_DAYS).fillna(0.0)
    corr = window.corr().to_numpy()
    adj = (corr >= CORR_THRESHOLD).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    return adj


def build_adjacency() -> tuple[np.ndarray, pd.DataFrame]:
    """Build the sector adjacency matrix and the ticker-to-sector map.

    Returns:
        (adjacency, mapping) where adjacency is (N, N) float32 and
        mapping is a DataFrame with columns (ticker, sector).
    """
    sector_df = _load_sector_map()
    null_frac = sector_df["sector"].isna().mean()
    tickers = sector_df["ticker"].tolist()
    n = len(tickers)

    if null_frac > NULL_TOLERANCE:
        print(
            f"[build_sector_adjacency] WARNING: {null_frac:.2%} of "
            f"tickers have null GICS sector; falling back to 60-day "
            f"rolling correlation thresholded at {CORR_THRESHOLD}.",
            flush=True,
        )
        adj = _fallback_correlation_adjacency(tickers)
    else:
        sectors = sector_df["sector"].fillna("Unknown").to_numpy()
        # Vectorised same-sector outer comparison.
        adj = (sectors[:, None] == sectors[None, :]).astype(np.float32)
        np.fill_diagonal(adj, 1.0)

    return adj, sector_df


def main() -> None:
    """Build, save, and summarise the sector adjacency artefacts."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adj, sector_df = build_adjacency()
    n = adj.shape[0]
    np.save(ADJACENCY_OUT, adj.astype(np.float32))
    sector_df.to_csv(SECTOR_MAP_OUT, index=False)

    print(f"[build_sector_adjacency] tickers: {n}")
    print(f"[build_sector_adjacency] adjacency shape: {adj.shape}")
    off_diag = adj.sum() - np.trace(adj)
    sparsity = 1.0 - (off_diag / (n * n - n))
    print(
        f"[build_sector_adjacency] off-diagonal sparsity: "
        f"{sparsity:.4f} (fraction of zero entries)"
    )
    print(f"[build_sector_adjacency] mean degree: {adj.sum(axis=1).mean():.2f}")
    print("[build_sector_adjacency] sector counts:")
    counts = sector_df["sector"].fillna("Unknown").value_counts()
    for sector, count in counts.items():
        print(f"  {sector:<28s} {count}")
    print(f"[build_sector_adjacency] saved adjacency to {ADJACENCY_OUT}")
    print(f"[build_sector_adjacency] saved sector map to {SECTOR_MAP_OUT}")


if __name__ == "__main__":
    main()
