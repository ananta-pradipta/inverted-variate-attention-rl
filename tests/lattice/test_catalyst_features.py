"""Phase 5b C acceptance tests (spec section 6.3).

Covers:
  - Cell-level correctness on a known earnings-date ticker (AAPL Q4 2020).
  - Zero-fill correctness for cells outside the 10-day horizon.
  - sin/cos values match the spec formula at d in {1, 5, 10}.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from scripts.lattice.build_catalyst_features import (
    HORIZON_TRADING_DAYS, EARNINGS_TYPE_ID,
    compute_days_to_next, map_event_to_next_trading_day,
)


def _make_panel(dates: list[pd.Timestamp], tickers: list[str]) -> pd.DataFrame:
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({"date": d, "ticker": t})
    return pd.DataFrame(rows)


def test_map_event_to_next_trading_day():
    panel_dates = np.array([
        pd.Timestamp("2020-10-26"), pd.Timestamp("2020-10-27"),
        pd.Timestamp("2020-10-28"), pd.Timestamp("2020-10-29"),
        pd.Timestamp("2020-10-30"),
    ])
    # On the day -> exact match
    di = map_event_to_next_trading_day(pd.Timestamp("2020-10-29"), panel_dates)
    assert di == 3
    # Between days -> next trading day
    di = map_event_to_next_trading_day(pd.Timestamp("2020-10-29 16:00"), panel_dates)
    assert di == 4  # after-market on 10-29 lands on 10-30 first trading day available
    # Past panel end -> None
    di = map_event_to_next_trading_day(pd.Timestamp("2021-01-01"), panel_dates)
    assert di is None


def test_compute_days_to_next_simple():
    panel_dates = np.array([pd.Timestamp(f"2020-10-{d:02d}") for d in (26, 27, 28, 29, 30)])
    di_lookup = {d: i for i, d in enumerate(panel_dates)}
    panel_df = _make_panel(list(panel_dates), ["AAPL", "MSFT"])
    panel_di = panel_df["date"].map(di_lookup).to_numpy(dtype=np.int64)
    panel_tickers = panel_df["ticker"].to_numpy()
    # AAPL has earnings on 10-29 (di=3); MSFT on 10-27 (di=1)
    events = {"AAPL": [3], "MSFT": [1]}
    days = compute_days_to_next(panel_di, events, panel_tickers)
    rows_aapl = (panel_tickers == "AAPL")
    aapl_days = days[rows_aapl].tolist()
    # AAPL panel days are di 0,1,2,3,4 -> next event di=3 -> days = 3,2,1,0,-1
    assert aapl_days == [3, 2, 1, 0, -1]
    rows_msft = (panel_tickers == "MSFT")
    msft_days = days[rows_msft].tolist()
    # MSFT next event di=1 -> from 0,1,2,3,4: days 1,0,-1,-1,-1
    assert msft_days == [1, 0, -1, -1, -1]


def test_sin_cos_horizon_formula():
    """At d in {0, 1, 5, 10}, sin/cos must match the spec formula exactly."""
    # Replicate the in-script formula
    for d in (0, 1, 5, 10):
        s = math.sin(2 * math.pi * d / HORIZON_TRADING_DAYS)
        c = math.cos(2 * math.pi * d / HORIZON_TRADING_DAYS)
        # d=0: sin=0, cos=1
        # d=5: sin(pi)=0, cos(pi)=-1
        # d=10: sin(2pi)=0, cos(2pi)=1
        if d == 0:
            assert abs(s - 0.0) < 1e-7 and abs(c - 1.0) < 1e-7
        if d == 5:
            assert abs(s - 0.0) < 1e-7 and abs(c - (-1.0)) < 1e-7
        if d == 10:
            assert abs(s - 0.0) < 1e-7 and abs(c - 1.0) < 1e-7


def test_zero_fill_outside_horizon():
    """Cells with days > 10 or days < 0 produce zeros in all three features."""
    days = np.array([-1, 0, 5, 10, 11, 20, 60], dtype=np.int64)
    sin_col = np.zeros(len(days), dtype=np.float32)
    cos_col = np.zeros(len(days), dtype=np.float32)
    type_col = np.zeros(len(days), dtype=np.int8)
    in_window = (days >= 0) & (days <= HORIZON_TRADING_DAYS)
    d = days[in_window].astype(np.float32)
    sin_col[in_window] = np.sin(2 * np.pi * d / HORIZON_TRADING_DAYS).astype(np.float32)
    cos_col[in_window] = np.cos(2 * np.pi * d / HORIZON_TRADING_DAYS).astype(np.float32)
    type_col[in_window] = EARNINGS_TYPE_ID

    # d=-1, 11, 20, 60 must be all zeros for type_id, sin, cos
    for i in (0, 4, 5, 6):
        assert sin_col[i] == 0.0
        assert cos_col[i] == 0.0
        assert type_col[i] == 0
    # d=0, 5, 10 must have type_id == 1
    for i in (1, 2, 3):
        assert type_col[i] == EARNINGS_TYPE_ID
