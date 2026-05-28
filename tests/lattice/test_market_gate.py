"""Unit tests for src/invar/modules/market_gate.py (InVAR v4 spec)."""
from __future__ import annotations

import torch

from src.invar.modules.market_gate import MarketGate


def test_identity_at_init():
    g = MarketGate(num_features=26, market_dim=24)
    x = torch.randn(2, 10, 60, 26)
    m = torch.randn(2, 24)
    x_tilde, alpha = g(x, m)
    assert torch.allclose(alpha.mean(dim=-1), torch.ones(2), atol=1e-4)
    assert alpha.std(dim=-1).max() < 1e-4
    assert torch.allclose(x_tilde, x, atol=1e-5)


def test_sum_to_F():
    for hidden in [0, 32]:
        g = MarketGate(num_features=26, market_dim=24, hidden_dim=hidden)
        for p in g.parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, std=0.1)
        m = torch.randn(4, 24)
        _, alpha = g(torch.zeros(4, 1, 1, 26), m)
        assert torch.allclose(alpha.sum(dim=-1), torch.full((4,), 26.0), atol=1e-3)


def test_beta_sensitivity():
    g_lo = MarketGate(num_features=26, market_dim=24, beta_init=0.5)
    g_hi = MarketGate(num_features=26, market_dim=24, beta_init=8.0)
    for g in (g_lo, g_hi):
        for p in g.parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, std=1.0)
    m = torch.randn(1, 24)
    _, a_lo = g_lo(torch.zeros(1, 1, 1, 26), m)
    _, a_hi = g_hi(torch.zeros(1, 1, 1, 26), m)
    p_lo = a_lo[0] / 26.0
    p_hi = a_hi[0] / 26.0
    kl_lo = (p_lo * (p_lo.log() - torch.log(torch.tensor(1.0 / 26)))).sum()
    kl_hi = (p_hi * (p_hi.log() - torch.log(torch.tensor(1.0 / 26)))).sum()
    assert kl_lo > kl_hi


def test_backprop():
    g = MarketGate(num_features=26, market_dim=24)
    x = torch.randn(2, 10, 60, 26)
    m = torch.randn(2, 24)
    x_tilde, _ = g(x, m)
    loss = x_tilde.pow(2).mean()
    loss.backward()
    if isinstance(g.proj, torch.nn.Linear):
        assert g.proj.weight.grad is not None
        assert g.proj.weight.grad.abs().sum() > 0
    assert g.log_beta.grad is not None


def test_shape_and_macro_seq():
    g = MarketGate(num_features=26, market_dim=24)
    x = torch.randn(2, 10, 60, 26)
    m_seq = torch.randn(2, 60, 24)
    x_tilde, alpha = g(x, m_seq)
    assert x_tilde.shape == (2, 10, 60, 26)
    assert alpha.shape == (2, 26)
