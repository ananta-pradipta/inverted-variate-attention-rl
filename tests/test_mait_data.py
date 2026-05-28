"""Phase 1 acceptance tests for MAiT data adapter.

Three assertions per the implementation prompt Phase 1 deliverable:
  1. x_panel shape is (N_t, 24, 60).
  2. x_macro_lookback shape is (n_macro, 60) where n_macro is 17 (because
     market_breadth_proxy is 100 percent non-null on 2026-05-11) or 16 if
     ever excluded.
  3. regime_input is a 5-vector with finite values, slice positions
     corresponding to [vix, vix_term_slope, slope_2s10s, hyg_5d_ret,
     market_breadth_proxy].
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.invar.baselines.mait_data import (
    KEEP,
    N_MACRO_KEPT,
    N_PANEL_KEPT,
    REGIME_DIM,
    MaitBatchAdapter,
)
from src.invar.data.dataset import InvarDataset


def _fold1_train_adapter():
    ds = InvarDataset(fold=1, split="train")
    adapter = MaitBatchAdapter(ds, ensure_scaler_persisted=True)
    return ds, adapter


def test_x_panel_shape() -> None:
    ds, adapter = _fold1_train_adapter()
    batch = ds.get(int(ds._eligible_idx[0]))
    out = adapter.adapt(batch)
    n_t = batch.features.shape[0]
    assert out.x_panel.shape == (n_t, 24, 60), (
        f"expected (N_t={n_t}, 24, 60), got {tuple(out.x_panel.shape)}"
    )
    assert N_PANEL_KEPT == 24
    # Confirm dropped names
    assert "catalyst_type_id" not in KEEP["kept_panel_names"]
    assert "has_stocktwits" not in KEEP["kept_panel_names"]


def test_x_macro_lookback_shape() -> None:
    ds, adapter = _fold1_train_adapter()
    batch = ds.get(int(ds._eligible_idx[0]))
    out = adapter.adapt(batch)
    # 2026-05-11 pivot: 8-feature minimal regime-discriminating subset
    # (was 17 earlier today, dropped to 8 per docs/macro_feature_analysis.md
    # after the 17-feature MAiT F2 5-seed mean came in at -0.0064).
    assert N_MACRO_KEPT == 8, N_MACRO_KEPT
    assert out.x_macro_lookback.shape == (N_MACRO_KEPT, 60), (
        f"expected ({N_MACRO_KEPT}, 60), got {tuple(out.x_macro_lookback.shape)}"
    )
    # The 8 kept features.
    expected_keep = {
        "vix", "vix_term_slope", "move_proxy",
        "dgs2", "dgs10", "slope_2s10s", "breakeven_10y",
        "hyg_5d_ret",
    }
    assert set(KEEP["kept_macro_names"]) == expected_keep, (
        f"kept macro mismatch: got {set(KEEP['kept_macro_names'])}"
    )


def test_regime_input_5vec_finite_and_correct_slice() -> None:
    ds, adapter = _fold1_train_adapter()
    batch = ds.get(int(ds._eligible_idx[0]))
    out = adapter.adapt(batch)
    assert out.regime_input.shape == (5,), out.regime_input.shape
    assert REGIME_DIM == 5
    assert torch.all(torch.isfinite(out.regime_input)), (
        f"regime_input has non-finite values: {out.regime_input}"
    )
    # Identity check: each regime slot must equal the query-day value of
    # the same feature pulled directly from x_macro_lookback at the
    # corresponding position in kept_macro_names. Slot 5 was
    # market_breadth_proxy in the 17-feature preset; swapped to
    # breakeven_10y in the 8-feature minimal preset (2026-05-11 pivot)
    # because it is the F2 regime discriminator and market_breadth_proxy
    # is dropped from the minimal set.
    expected_names = ["vix", "vix_term_slope", "slope_2s10s",
                      "hyg_5d_ret", "breakeven_10y"]
    for k, name in enumerate(expected_names):
        assert name in KEEP["kept_macro_names"], (
            f"regime slot {k} feature {name} missing from kept macro list"
        )
        pos = KEEP["kept_macro_names"].index(name)
        expected_val = out.x_macro_lookback[pos, -1].item()
        got_val = out.regime_input[k].item()
        assert abs(expected_val - got_val) < 1.0e-6, (
            f"regime slot {k} ({name}) = {got_val} but "
            f"x_macro_lookback at kept position {pos}, t=-1 = {expected_val}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
