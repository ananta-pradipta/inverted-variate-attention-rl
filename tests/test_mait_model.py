"""Phase 2 acceptance tests for MAiT.

Five tests per the implementation prompt Phase 2 deliverable:
  1. Forward pass shapes: y_hat (16,), s_panel (16,), s_macro (16,), g scalar.
  2. Gate in [0, 1] for 100 random regime inputs.
  3. Backward pass produces non-zero gradients on gate MLP weights.
  4. With g forced to 0, y_hat == s_panel within 1e-6 (Stream A fallback).
  5. Parameter count logged. Soft target < 1.2M, hard cap 1.5M.
"""
from __future__ import annotations

import pytest
import torch

from src.invar.baselines.mait import MAiT, count_parameters, mait_loss


def _build_model() -> MAiT:
    return MAiT(
        n_panel=24, n_macro=17, L_lookback=60,
        d_model=128, n_heads=4, d_ff=256, n_layers=3,
        dropout=0.1, stream_dropout_p=0.15, regime_dim=5,
    )


def test_forward_shapes() -> None:
    torch.manual_seed(0)
    model = _build_model().eval()
    B = 16
    x_panel = torch.randn(B, 24, 60)
    x_macro_lookback = torch.randn(17, 60)
    regime_input = torch.randn(5)
    y_hat, s_panel, s_macro, g = model(
        x_panel, x_macro_lookback, regime_input, train_mode=False,
    )
    assert y_hat.shape == (B,)
    assert s_panel.shape == (B,)
    assert s_macro.shape == (B,)
    # g is a scalar tensor; allow shape () or shape (1,).
    assert g.dim() == 0 or (g.dim() == 1 and g.numel() == 1), g.shape


def test_gate_in_unit_interval() -> None:
    torch.manual_seed(1)
    model = _build_model().eval()
    B = 8
    x_panel = torch.randn(B, 24, 60)
    x_macro_lookback = torch.randn(17, 60)
    gs = []
    for _ in range(100):
        regime_input = torch.randn(5)
        _, _, _, g = model(
            x_panel, x_macro_lookback, regime_input, train_mode=False,
        )
        gs.append(g.item())
    gs = torch.tensor(gs)
    assert torch.all(gs >= 0.0), f"min gate {gs.min().item()}"
    assert torch.all(gs <= 1.0), f"max gate {gs.max().item()}"


def test_backward_yields_nonzero_gate_grads() -> None:
    torch.manual_seed(2)
    model = _build_model().train()
    # Disable stream-dropout for this test (we want a deterministic path
    # through the gate so the backward is clean).
    model.stream_dropout_p = 0.0

    B = 16
    x_panel = torch.randn(B, 24, 60, requires_grad=False)
    x_macro_lookback = torch.randn(17, 60, requires_grad=False)
    regime_input = torch.randn(5, requires_grad=False)
    y_true = torch.randn(B)

    y_hat, s_panel, s_macro, g = model(
        x_panel, x_macro_lookback, regime_input, train_mode=True,
    )
    loss = mait_loss(y_hat, s_panel, g, y_true)
    loss.backward()

    nonzero_grads = []
    for name, p in model.gate_mlp.named_parameters():
        assert p.grad is not None, f"gate_mlp.{name} has no grad"
        gnorm = p.grad.detach().abs().sum().item()
        nonzero_grads.append((name, gnorm))
    print(f"gate_mlp grad norms (sum |grad|): {nonzero_grads}")
    assert any(g_norm > 0.0 for _, g_norm in nonzero_grads), (
        f"all gate_mlp grads are zero: {nonzero_grads}"
    )


def test_stream_a_fallback_at_g_zero() -> None:
    torch.manual_seed(3)
    model = _build_model().eval()
    B = 16
    x_panel = torch.randn(B, 24, 60)
    x_macro_lookback = torch.randn(17, 60)
    regime_input = torch.randn(5)
    y_hat, s_panel, s_macro, g = model(
        x_panel, x_macro_lookback, regime_input,
        train_mode=False, force_g=0.0,
    )
    assert g.item() == 0.0
    delta = (y_hat - s_panel).abs().max().item()
    assert delta < 1.0e-6, (
        f"Stream A fallback delta {delta} exceeds 1e-6 at g = 0; "
        f"y_hat must equal s_panel mechanically when the gate is closed."
    )


def test_parameter_count() -> None:
    model = _build_model()
    n_params = count_parameters(model)
    print(f"MAiT total trainable parameters: {n_params:,}")
    # Hard cap: stop if above 1.5M.
    assert n_params < 1_500_000, (
        f"MAiT parameter count {n_params:,} above 1.5M hard cap; "
        f"something is off in the architecture."
    )
    if n_params >= 1_200_000:
        # Soft target exceeded but not hard cap; let the test pass with a
        # warning so the deliverable can flag the deviation.
        pytest.skip(
            f"Soft target (under 1.2M) exceeded at {n_params:,}; "
            f"under hard cap, continuing with deviation flag.",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
