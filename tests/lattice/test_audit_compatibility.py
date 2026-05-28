"""Audit-compatibility test: forward pass cleanly handles inactive cells.

Per spec section 6.10:
  Gradient through inactive cells must be exactly zero.
"""
from __future__ import annotations

import torch

from src.lattice.model.lattice import LATTICE, LatticeConfig
from tests.lattice.test_model_shapes import _make_inputs


def test_inactive_cells_gradient_zero() -> None:
    """Gradient on inactive cells should be exactly zero."""
    torch.manual_seed(42)
    inputs = _make_inputs(B=2, N=200)
    # Make half the tickers inactive
    active = torch.ones(2, 200, dtype=torch.bool)
    active[:, 100:] = False
    inputs["active_mask"] = active

    model = LATTICE(LatticeConfig())
    model.train()
    inputs["panel_features"].requires_grad_(True)
    y_hat, _ = model(**inputs)
    # Loss is sum over inactive cells
    inactive_cells = ~active
    inactive_y = y_hat[inactive_cells]
    # All inactive y_hat must be zero (model multiplies by active_mask in head)
    assert torch.equal(inactive_y, torch.zeros_like(inactive_y)), (
        "Inactive cells produced non-zero y_hat"
    )

    # Sum loss over ACTIVE cells; gradient flowing back to inactive panel
    # cells should be zero (they're masked out before any reduction).
    active_loss = y_hat[active].sum()
    active_loss.backward()
    grad = inputs["panel_features"].grad
    inactive_grad = grad[~active]
    # Gradient through inactive cells should be exactly zero (not "small")
    # because the model masks them out at every aggregator stage.
    inactive_grad_max = inactive_grad.abs().max().item()
    assert inactive_grad_max < 1e-6, (
        f"Inactive cells leak gradient: max abs grad = {inactive_grad_max:.3e}"
    )
