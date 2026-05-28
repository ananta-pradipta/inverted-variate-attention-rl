"""Contract-conformance tests for the real S&P 500 panel loader.

These run only where the data artifacts are present (Wulver, or a local
checkout with the parquet files). Set TLI_SP500_DATA_ROOT to the directory
containing ``lattice/`` and ``raw/``; otherwise the module is skipped so the
suite still passes where the data is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

_ROOT = os.environ.get(
    "TLI_SP500_DATA_ROOT", str(Path.home() / "phd-research" / "data")
)
_HAVE = (
    Path(_ROOT) / "lattice/processed/panel_features.parquet"
).is_file()

pytestmark = pytest.mark.skipif(
    not _HAVE, reason=f"S&P 500 panel artifacts not found under {_ROOT}"
)

if _HAVE:
    from src.data.sp500_panel import FEATURE_COLUMNS, SP500Panel


def _panel() -> "SP500Panel":
    return SP500Panel(_ROOT, lookback=20, label_horizon=5,
                      train_end_index=995)


def test_dimensions_and_calendar() -> None:
    p = _panel()
    assert p.lookback == 20
    assert p.label_horizon == 5
    assert p.n_features == len(FEATURE_COLUMNS) == 26
    assert p.macro_dim == 24
    assert len(p.trading_days()) > 1500


def test_contract_shapes_consistent() -> None:
    p = _panel()
    day = 1200
    tickers, feats = p.feature_window(day)
    n = len(tickers)
    assert n > 50  # a broad cross-section
    assert feats.shape == (n, 20, 26)
    assert np.isfinite(feats).all()  # standardised + NaN-filled
    assert p.macro_vector(day).shape == (24,)
    assert np.isfinite(p.macro_vector(day)).all()
    raw, z = p.forward_label(day)
    mask = p.tradable_mask(day)
    assert raw.shape == z.shape == mask.shape == (n,)
    assert mask.dtype == bool


def test_within_day_label_is_zscored() -> None:
    p = _panel()
    raw, z = p.forward_label(1200)
    mask = p.tradable_mask(1200)
    v = z[mask & np.isfinite(z)]
    assert v.size >= 2
    assert abs(float(v.mean())) < 1e-6
    assert abs(float(v.std()) - 1.0) < 1e-6


def test_realized_returns_and_end_masking() -> None:
    p = _panel()
    n = len(p.active_tickers(1000))
    _, r1 = p.realized_returns(1000, 1)
    _, r5 = p.realized_returns(1000, 5)
    assert r1.shape == (n,) and r5.shape == (n,)
    last = len(p.trading_days()) - 1
    assert not p.tradable_mask(last).any()


def test_train_only_standardiser_changes_with_train_end() -> None:
    a = SP500Panel(_ROOT, 20, 5, train_end_index=500)
    b = SP500Panel(_ROOT, 20, 5, train_end_index=1500)
    # Different training windows -> different standardisation stats.
    assert not np.allclose(a._f_mean, b._f_mean)
