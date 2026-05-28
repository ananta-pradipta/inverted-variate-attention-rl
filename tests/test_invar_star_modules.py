"""Unit tests for InVAR-STAR modules.

Tests are organized to match Phase 1 acceptance criteria in the implementation
spec (10 tests total, including the load-bearing architectural-collapse test
in Strand A of design doc Section 5).
"""
from __future__ import annotations

import math

import pytest
import torch

from src.invar_star.model import (
    set_global_seed,
    MacroVariateBank,
    SelfThrottlingGate,
    ThrottledVariateAttention,
    InVARSTARBlock,
    MoERankingHead,
    InVARSTAR,
)
from src.invar_star.losses import (
    throttle_kl_prior,
    load_balance_loss,
    weighted_pearson_ic_loss,
)


# Test 1: set_global_seed is deterministic.
def test_set_global_seed_deterministic() -> None:
    set_global_seed(42)
    a = torch.randn(4)
    set_global_seed(42)
    b = torch.randn(4)
    assert torch.allclose(a, b)


# Test 2: MacroVariateBank produces (B, 50, d_model) from (B, 26, L) + (B, 24, L).
def test_macro_variate_bank_shape() -> None:
    set_global_seed(42)
    bank = MacroVariateBank(lookback=60, n_stock_feats=26,
                            n_macro_feats=24, d_model=128)
    B = 8
    x_s = torch.randn(B, 26, 60)
    x_m = torch.randn(B, 24, 60)
    out = bank(x_s, x_m)
    assert out.shape == (B, 50, 128)


# Test 3: SelfThrottlingGate is bimodal under low tau in training, deterministic at eval.
def test_self_throttling_gate_concrete_relaxation() -> None:
    set_global_seed(42)
    gate = SelfThrottlingGate(phi_dim=64)
    phi = torch.zeros(10_000, 64)
    beta_train = gate(phi, tau=0.01, training=True).detach()
    # Empirical distribution near-bimodal at 0 and 1 (mass split across the
    # two extremes). Check by counting samples in the middle bin.
    middle = ((beta_train > 0.2) & (beta_train < 0.8)).float().mean().item()
    assert middle < 0.05, f"expected near-bimodal, middle mass {middle}"
    # Eval is deterministic.
    beta_eval_1 = gate(phi[:32], tau=0.01, training=False)
    beta_eval_2 = gate(phi[:32], tau=0.01, training=False)
    assert torch.allclose(beta_eval_1, beta_eval_2)


# Test 4 (LOAD-BEARING): architectural-collapse — beta=1e-8 reproduces
# stock-only attention output to within 1e-5.
def test_architectural_collapse_under_zero_beta() -> None:
    set_global_seed(42)
    d_model, n_heads, B, n_stock, n_macro = 128, 4, 4, 26, 24
    attn = ThrottledVariateAttention(d_model, n_heads, n_stock, n_macro)
    attn.eval()
    h_full = torch.randn(B, n_stock + n_macro, d_model)
    h_stock_only = h_full[:, :n_stock].clone()

    # Pass 1: full bank with beta -> 0. Read only the stock-row outputs.
    beta = torch.full((B, 1), 1.0e-8)
    with torch.no_grad():
        out_full = attn(h_full, beta)
    out_stock_from_full = out_full[:, :n_stock]

    # Pass 2: stock-only attention via the SAME projection weights so the
    # comparison is byte-equivalent. Build a stock-only forward by hand using
    # the same Q, K, V projections, with no macro-side bias.
    with torch.no_grad():
        B_, N_stock, D = h_stock_only.shape
        qkv = attn.qkv(h_stock_only).reshape(
            B_, N_stock, 3, attn.n_heads, attn.d_head,
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn.d_head)
        attn_w = torch.softmax(scores, dim=-1)
        out_stock_only = (attn_w @ v).transpose(1, 2).reshape(B_, N_stock, D)
        out_stock_only = attn.out(out_stock_only)

    delta = (out_stock_from_full - out_stock_only).abs().max().item()
    assert delta < 1.0e-5, (
        f"architectural-collapse delta {delta} exceeds 1e-5 tolerance; "
        f"InVAR-STAR cannot recover vanilla iTransformer behaviour."
    )


# Test 5: InVARSTARBlock preserves shape.
def test_invar_star_block_shape() -> None:
    set_global_seed(42)
    block = InVARSTARBlock(d_model=128, n_heads=4)
    block.eval()
    h = torch.randn(8, 50, 128)
    beta = torch.full((8, 1), 0.5)
    out = block(h, beta)
    assert out.shape == h.shape


# Test 6 (LOAD-BEARING): router-decoupling — when beta=0, router input
# Jacobian wrt phi is below 1e-6.
def test_moe_router_decouples_from_phi_when_beta_zero() -> None:
    set_global_seed(42)
    d_model, phi_dim, B = 128, 64, 4
    head = MoERankingHead(d_model, phi_dim, n_experts=4, top_k=2,
                          noise_std=0.0, expert_dropout=0.0)
    head.eval()
    z = torch.randn(B, d_model)
    phi = torch.randn(B, phi_dim, requires_grad=True)
    beta = torch.zeros(B, 1)
    y_hat, route_probs = head(z, phi, beta, training=False)
    # Sum the route_probs (scalar) and differentiate wrt phi; norm should be
    # near zero because the router input has a `beta * macro_proj(phi)` term
    # that vanishes when beta is zero, and z is detached from phi.
    grad = torch.autograd.grad(route_probs.sum(), phi, retain_graph=False)[0]
    norm = grad.abs().max().item()
    assert norm < 1.0e-6, f"router Jacobian wrt phi at beta=0 is {norm}, expected near 0"


# Test 7: full InVARSTAR forward returns expected shapes.
def test_invar_star_full_forward_shapes() -> None:
    set_global_seed(42)
    model = InVARSTAR(lookback=60, d_model=128, n_heads=4, n_layers=3,
                      n_experts=4, top_k=2, n_stock=26, n_macro=24)
    model.eval()
    B = 5
    x_s = torch.randn(B, 26, 60)
    x_m = torch.randn(B, 24, 60)
    out = model(x_s, x_m, tau=1.0)
    assert out["y_hat"].shape == (B, 1)
    assert out["beta"].shape == (B, 1)
    assert out["route_probs"].shape == (B, 4)
    assert torch.all(out["beta"] > 0.0) and torch.all(out["beta"] < 1.0)


# Test 8: throttle_kl_prior is non-negative and lower for Beta(0.4, 0.6)
# samples than for unimodal-at-0.5 samples. The softmax-over-bins
# approximation of Beta(0.4, 0.6) is moderately (not extremely) bimodal,
# so samples that exactly match the prior shape should score lowest.
def test_throttle_kl_prior_prefers_target_distribution() -> None:
    set_global_seed(42)
    from torch.distributions import Beta as BetaDist
    # Samples from the target Beta(0.4, 0.6) prior.
    beta_target = BetaDist(torch.tensor(0.4), torch.tensor(0.6)).sample((1024, 1))
    kl_target = throttle_kl_prior(beta_target)
    # Samples from a unimodal-at-0.5 distribution (Beta(5, 5)) that
    # concentrates in the middle bins, mismatching the bimodal prior.
    beta_unimodal = BetaDist(torch.tensor(5.0), torch.tensor(5.0)).sample((1024, 1))
    kl_unimodal = throttle_kl_prior(beta_unimodal)
    assert kl_target >= 0.0 and kl_unimodal >= 0.0
    assert kl_target < kl_unimodal, (
        f"expected target-distribution KL ({kl_target}) < "
        f"unimodal KL ({kl_unimodal})"
    )


# Test 9: load_balance_loss is positive and minimised at uniform routing.
def test_load_balance_loss_basic() -> None:
    set_global_seed(42)
    K = 4
    B = 256
    # Uniform routing: each expert wins ~25 percent of tokens.
    uniform = torch.ones(B, K) / K
    lb_uniform = load_balance_loss(uniform)
    # Collapsed routing: all tokens go to expert 0.
    collapsed = torch.zeros(B, K)
    collapsed[:, 0] = 1.0
    lb_collapsed = load_balance_loss(collapsed)
    assert lb_uniform >= 0.0
    assert lb_collapsed > lb_uniform, (
        f"collapsed lb ({lb_collapsed}) should exceed uniform lb ({lb_uniform})"
    )


# Test 10: weighted_pearson_ic_loss matches the analytic Pearson correlation.
def test_weighted_pearson_ic_loss_matches_pearson() -> None:
    set_global_seed(42)
    y = torch.randn(64)
    y_hat = 0.5 * y + 0.3 * torch.randn(64)
    loss = weighted_pearson_ic_loss(y_hat, y).item()
    # Direct Pearson computation as ground truth.
    a = y_hat - y_hat.mean()
    b = y - y.mean()
    r = (a * b).sum() / (a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt())
    assert abs(loss - (-r.item())) < 1.0e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
