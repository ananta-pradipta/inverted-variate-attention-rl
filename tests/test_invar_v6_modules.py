"""Unit tests for InVAR-v6 modules.

Covers MarketGateV2, MacroWindowEncoder, and DynamicBankController.
Regime labels and calibration tests are deferred to a follow-up file
because they require disk I/O (the parquet cache).
"""
from __future__ import annotations

import torch

from src.invar.modules.dynamic_bank_controller import DynamicBankController
from src.invar.modules.macro_window_encoder import MacroWindowEncoder
from src.invar.modules.market_gate_v2 import MarketGateV2


# MarketGateV2

def test_market_gate_v2_identity_at_init() -> None:
    gate = MarketGateV2(
        num_features=26, macro_state_dim=64, gate_form="softmax_F",
        hidden_dim=0, identity_init=True,
    )
    x = torch.randn(2, 50, 60, 26)
    m = torch.randn(2, 64)
    x_out, alpha = gate(x, m)
    assert torch.allclose(alpha, torch.ones_like(alpha), atol=1.0e-4)
    assert torch.allclose(x_out, x, atol=1.0e-4)


def test_market_gate_v2_softmax_sums_to_F() -> None:
    gate = MarketGateV2(
        num_features=26, macro_state_dim=64, gate_form="softmax_F",
        identity_init=False,
    )
    m = torch.randn(3, 64)
    x = torch.randn(3, 10, 60, 26)
    _, alpha = gate(x, m)
    assert torch.allclose(alpha.sum(dim=-1), torch.full((3,), 26.0), atol=1.0e-3)


def test_market_gate_v2_sigmoid_centered_mean_one() -> None:
    gate = MarketGateV2(
        num_features=26, macro_state_dim=64, gate_form="sigmoid_centered",
        identity_init=False,
    )
    m = torch.randn(4, 64)
    x = torch.randn(4, 8, 60, 26)
    _, alpha = gate(x, m)
    assert torch.allclose(alpha.mean(dim=-1), torch.ones(4), atol=1.0e-4)


def test_market_gate_v2_sigmoid_residual_in_range() -> None:
    gate = MarketGateV2(
        num_features=26, macro_state_dim=64, gate_form="sigmoid_residual",
        residual_scale=0.25, identity_init=False,
    )
    m = torch.randn(2, 64) * 100.0  # extreme inputs to saturate tanh
    x = torch.randn(2, 5, 60, 26)
    _, alpha = gate(x, m)
    assert ((alpha >= 0.75 - 1.0e-3) & (alpha <= 1.25 + 1.0e-3)).all()


def test_market_gate_v2_beta_clamp() -> None:
    gate = MarketGateV2(
        num_features=26, macro_state_dim=64, beta_init=1000.0, learn_beta=True,
    )
    assert gate.beta.item() <= 20.0


def test_market_gate_v2_backprop() -> None:
    gate = MarketGateV2(num_features=8, macro_state_dim=4, identity_init=False)
    x = torch.randn(2, 3, 60, 8, requires_grad=True)
    m = torch.randn(2, 4, requires_grad=True)
    x_out, _ = gate(x, m)
    x_out.sum().backward()
    assert m.grad is not None


# MacroWindowEncoder

def test_macro_window_encoder_modes_shape() -> None:
    for mode in ("last", "mlp_flat", "temporal_attn", "gru"):
        enc = MacroWindowEncoder(
            macro_dim=24, lookback=60, out_dim=64, hidden_dim=128,
            mode=mode,
        )
        m = torch.randn(2, 60, 24)
        out = enc(m)
        assert out.shape == (2, 64), f"mode {mode}: got {out.shape}"


def test_macro_window_encoder_no_nan() -> None:
    enc = MacroWindowEncoder(mode="mlp_flat")
    m = torch.randn(3, 60, 24)
    out = enc(m)
    assert torch.isfinite(out).all()


def test_macro_window_encoder_unbatched() -> None:
    enc = MacroWindowEncoder(mode="last")
    m = torch.randn(60, 24)  # no batch dim
    out = enc(m)
    assert out.shape == (1, 64)


# DynamicBankController

def _bank_stats(B: int) -> dict:
    return {
        "retrieval_distance_z": torch.zeros(B),
        "retrieval_entropy_z": torch.zeros(B),
        "bank_value_norm_z": torch.zeros(B),
        "active_count_z": torch.zeros(B),
    }


def test_bank_controller_weight_shape_and_bounds() -> None:
    ctl = DynamicBankController(
        macro_state_dim=64, stats_dim=6, mode="hybrid",
        min_weight=0.05, max_weight=1.00,
    )
    macro = torch.randn(3, 64)
    stress = torch.randn(3, 6)
    w, _ = ctl(macro, _bank_stats(3), stress)
    assert w.shape == (3, 1)
    assert (w >= 0.05).all() and (w <= 1.00).all()


def test_bank_controller_deterministic_high_novelty_lowers_weight() -> None:
    ctl = DynamicBankController(mode="deterministic")
    macro = torch.zeros(2, 64)
    stress = torch.zeros(2, 6)
    bs_low = _bank_stats(2)
    bs_high = _bank_stats(2)
    bs_high["retrieval_distance_z"] = torch.tensor([3.0, 3.0])
    bs_high["retrieval_entropy_z"] = torch.tensor([3.0, 3.0])
    w_low, _ = ctl(macro, bs_low, stress)
    w_high, _ = ctl(macro, bs_high, stress)
    assert (w_high < w_low).all()


def test_bank_controller_deterministic_high_stress_raises_weight() -> None:
    ctl = DynamicBankController(mode="deterministic")
    macro = torch.zeros(2, 64)
    bs = _bank_stats(2)
    stress_low = torch.zeros(2, 6)
    stress_high = torch.full((2, 6), 3.0)
    w_low, _ = ctl(macro, bs, stress_low)
    w_high, _ = ctl(macro, bs, stress_high)
    assert (w_high > w_low).all()


def test_bank_controller_hybrid_backprop() -> None:
    ctl = DynamicBankController(mode="hybrid")
    macro = torch.randn(2, 64, requires_grad=True)
    stress = torch.randn(2, 6, requires_grad=True)
    w, _ = ctl(macro, _bank_stats(2), stress)
    w.sum().backward()
    assert macro.grad is not None


def test_bank_controller_deterministic_reproducible() -> None:
    ctl = DynamicBankController(mode="deterministic")
    macro = torch.randn(2, 64)
    stress = torch.randn(2, 6)
    w1, _ = ctl(macro, _bank_stats(2), stress)
    w2, _ = ctl(macro, _bank_stats(2), stress)
    assert torch.allclose(w1, w2)
