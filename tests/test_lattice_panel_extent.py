"""Smoke test: the lattice panel parquet on disk must cover the full
2015-01 to 2025-12 extended panel (>= 2700 unique trading days).

A silently truncated panel (e.g. the May-7 pre-extension build with
2003 days) causes Layer 1 / Ablation 6 checkpoint loads to fail with
a day_memory.mem_keys shape mismatch. This smoke test catches the
truncation at CI time rather than at training time.

If this test fails, re-run::

    invar_rl/scripts/wulver/sp500_panel_rebuild_pathb.sbatch

to rebuild from the extended raw inputs. See the runbook at
``reports/sp500/panel_rebuild_runbook_2026-05-23.md``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


PANEL_PATH = Path("data/lattice/processed/panel_features.parquet")
MIN_DAYS = 2700  # 2015-01-09 to 2025-12-31 is ~2755 trading days
MIN_TICKERS = 500
MIN_END_DATE = pd.Timestamp("2025-12-01")


@pytest.mark.skipif(
    not PANEL_PATH.exists(),
    reason=f"panel parquet missing at {PANEL_PATH}; build it via "
           "invar_rl/scripts/wulver/sp500_panel_rebuild_pathb.sbatch",
)
def test_panel_extent_2025() -> None:
    """Loaded panel covers the full 2015-2025 extended window."""
    panel = pd.read_parquet(PANEL_PATH, columns=["ticker", "date"])
    panel["date"] = pd.to_datetime(panel["date"])
    n_days = int(panel["date"].nunique())
    n_tickers = int(panel["ticker"].nunique())
    end_date = pd.Timestamp(panel["date"].max())

    assert n_days >= MIN_DAYS, (
        f"panel only has {n_days} unique trading days; expected "
        f">= {MIN_DAYS} (2015-01 to 2025-12 ~2755 days)"
    )
    assert n_tickers >= MIN_TICKERS, (
        f"panel only has {n_tickers} unique tickers; expected "
        f">= {MIN_TICKERS} for the S&P 500 universe"
    )
    assert end_date >= MIN_END_DATE, (
        f"panel ends at {end_date.date()}; expected to cover at "
        f"least {MIN_END_DATE.date()}"
    )
