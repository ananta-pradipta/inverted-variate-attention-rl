"""Shape tests for the LATTICE model.

Per spec section 6.10 acceptance gate:
  - pytest passes for batch_days in {1, 4, 16} and n_active_tickers in {200, 500, 505}.
"""
from __future__ import annotations

import pytest
import torch

from src.lattice.model.lattice import LATTICE, LatticeConfig


def _make_inputs(B: int, N: int, T: int = 60, F: int = 30, K_top: int = 8,
                  macro_dim: int = 24, regime_key_dim: int = 14,
                  novelty_key_dim: int = 8):
    """Build random inputs with the right shapes for a LATTICE forward pass."""
    panel_features = torch.randn(B, N, T, F)
    macro_state = torch.randn(B, macro_dim)
    size_decile = torch.randint(0, 10, (B, N))
    liquidity_decile = torch.randint(0, 10, (B, N))
    sector_id = torch.randint(0, 11, (B, N))
    age_bucket = torch.randint(0, 4, (B, N))
    regime_query_keys = torch.randn(B, regime_key_dim)
    novelty_query_keys = torch.randn(B, N, novelty_key_dim)
    novelty_sector_ids = torch.randint(0, 11, (B, N))
    active_mask = torch.ones(B, N, dtype=torch.bool)
    day_index = torch.arange(B, dtype=torch.long) + 100
    # correlation-graph neighbor indices: [B, N, K_top]; -1 is pad
    corr_neighbor_idx = torch.randint(0, N, (B, N, K_top))
    corr_neighbor_mask = torch.ones(B, N, K_top, dtype=torch.bool)
    return dict(
        panel_features=panel_features, macro_state=macro_state,
        cohort_size_decile=size_decile, cohort_liquidity_decile=liquidity_decile,
        cohort_sector_id=sector_id, cohort_age_bucket=age_bucket,
        regime_query_keys=regime_query_keys,
        novelty_query_keys=novelty_query_keys,
        novelty_sector_ids=novelty_sector_ids,
        active_mask=active_mask, day_index=day_index,
        corr_neighbor_idx=corr_neighbor_idx, corr_neighbor_mask=corr_neighbor_mask,
    )


@pytest.mark.parametrize("B,N", [(1, 200), (4, 500), (16, 505)])
def test_lattice_forward_shape(B: int, N: int) -> None:
    cfg = LatticeConfig()
    model = LATTICE(cfg)
    model.eval()
    inputs = _make_inputs(B, N)
    with torch.no_grad():
        y_hat, balance_loss = model(**inputs)
    assert y_hat.shape == (B, N), f"y_hat shape {y_hat.shape} != ({B}, {N})"
    assert balance_loss.dim() == 0, "balance_loss should be a scalar"
    assert torch.isfinite(y_hat).all(), "y_hat has non-finite entries"
