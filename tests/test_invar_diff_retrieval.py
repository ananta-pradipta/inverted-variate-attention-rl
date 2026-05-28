"""InVAR differentiable retrieval ablation acceptance tests."""
from __future__ import annotations

import torch

from src.invar.model.invar import (
    Invar, InvarConfig, RegimeAxisRetrieval,
)


def _cfg(retrieval_mode: str = "hard_topk") -> InvarConfig:
    return InvarConfig(
        n_features=26, lookback=60, macro_dim=24, d_model=128,
        n_heads=4, ffn_dim=256, n_layers=4, dropout=0.0,
        regime_axis="retrieval",
        bank_size=64, top_k_retrieve=32,
        use_market_gate=True,
        retrieval_mode=retrieval_mode,
    )


def _batch(N: int = 50) -> dict:
    torch.manual_seed(0)
    return dict(
        features=torch.randn(N, 60, 26),
        macro=torch.randn(60, 24),
        mask=torch.ones(N, dtype=torch.bool),
    )


def test_hard_topk_output_shape() -> None:
    cfg = _cfg("hard_topk")
    bank = RegimeAxisRetrieval(cfg)
    out = bank(torch.randn(128))
    assert out.shape == (32, 128)


def test_softmax_full_output_shape() -> None:
    cfg = _cfg("softmax_full")
    bank = RegimeAxisRetrieval(cfg)
    out = bank(torch.randn(128))
    assert out.shape == (64, 128)


def test_softmax_topk_output_shape() -> None:
    cfg = _cfg("softmax_topk")
    bank = RegimeAxisRetrieval(cfg)
    out = bank(torch.randn(128))
    assert out.shape == (32, 128)


def test_gumbel_topk_output_shape() -> None:
    cfg = _cfg("gumbel_topk")
    bank = RegimeAxisRetrieval(cfg)
    bank.train()
    out = bank(torch.randn(128))
    assert out.shape == (32, 128)


def test_softmax_full_grad_flows_to_all_keys() -> None:
    """softmax_full: backward through output produces grads on all 64 keys."""
    cfg = _cfg("softmax_full")
    bank = RegimeAxisRetrieval(cfg)
    bank.train()
    q = torch.randn(128)
    out = bank(q)
    out.sum().backward()
    assert bank.keys.grad is not None
    # Every key should have non-zero gradient (softmax over all 64 yields
    # a fully-connected gradient graph).
    nonzero_keys = (bank.keys.grad.abs().sum(dim=-1) > 0).long().sum().item()
    assert nonzero_keys == 64
    # Every value should also have a gradient.
    assert (bank.values.grad.abs().sum(dim=-1) > 0).long().sum().item() == 64


def test_softmax_topk_grad_flows_to_all_keys() -> None:
    """softmax_topk: keys all see gradient via the softmax denominator,
    values selectively at the top-K=32 positions."""
    cfg = _cfg("softmax_topk")
    bank = RegimeAxisRetrieval(cfg)
    bank.train()
    q = torch.randn(128)
    out = bank(q)
    out.sum().backward()
    nonzero_keys = (bank.keys.grad.abs().sum(dim=-1) > 0).long().sum().item()
    assert nonzero_keys == 64                                  # all keys
    # Values: only the top-K=32 should have non-zero gradient.
    nonzero_values = (bank.values.grad.abs().sum(dim=-1) > 0).long().sum().item()
    assert nonzero_values == 32


def test_hard_topk_grad_localised() -> None:
    """hard_topk: gradients should reach only the top-K=32 values."""
    cfg = _cfg("hard_topk")
    bank = RegimeAxisRetrieval(cfg)
    bank.train()
    q = torch.randn(128)
    out = bank(q)
    out.sum().backward()
    # Values: only 32 should have non-zero gradient.
    nonzero_values = (bank.values.grad.abs().sum(dim=-1) > 0).long().sum().item()
    assert nonzero_values == 32
    # Keys: hard topk does NOT propagate gradient through index selection,
    # so keys do NOT receive any gradient from the bank-only path.
    assert bank.keys.grad is None or bank.keys.grad.abs().sum().item() == 0


def test_gumbel_topk_grad_flows_to_all_keys() -> None:
    """gumbel_topk: gradient flows into all keys via the softmax(noisy logits)."""
    cfg = _cfg("gumbel_topk")
    bank = RegimeAxisRetrieval(cfg)
    bank.train()
    q = torch.randn(128)
    out = bank(q)
    out.sum().backward()
    nonzero_keys = (bank.keys.grad.abs().sum(dim=-1) > 0).long().sum().item()
    assert nonzero_keys == 64


def test_full_invar_forward_each_mode() -> None:
    """End-to-end Invar forward returns finite outputs for each mode."""
    for mode in ("hard_topk", "softmax_full", "softmax_topk", "gumbel_topk"):
        cfg = _cfg(mode)
        m = Invar(cfg)
        m.eval()
        out = m(**_batch(N=30))
        assert torch.isfinite(out["y_hat"]).all(), f"NaN y_hat in mode={mode}"
        assert torch.isfinite(out["regime_logits"]).all(), f"NaN regime_logits in mode={mode}"


def test_all_modes_one_train_step() -> None:
    """One forward+backward+AdamW step for each mode; finite loss + macro grad."""
    for mode in ("hard_topk", "softmax_full", "softmax_topk", "gumbel_topk"):
        cfg = _cfg(mode)
        m = Invar(cfg)
        m.train()
        optim = torch.optim.AdamW(m.parameters(), lr=1.0e-4)
        out = m(**_batch(N=80))
        loss = out["y_hat"].pow(2).mean()
        loss.backward()
        optim.step()
        # Macro encoder should always have grad.
        me_grad = sum(
            p.grad.abs().sum().item() for p in m.macro_encoder.parameters()
            if p.grad is not None
        )
        assert me_grad > 0.0, f"macro encoder has zero grad in mode={mode}"


def test_softmax_full_returns_normalized_weights() -> None:
    """softmax_full: cached scores should sum to 1."""
    cfg = _cfg("softmax_full")
    bank = RegimeAxisRetrieval(cfg)
    _ = bank(torch.randn(128))
    assert bank.last_top_scores is not None
    s = float(bank.last_top_scores.sum().item())
    assert abs(s - 1.0) < 1.0e-5


def test_modes_distinct_outputs() -> None:
    """Different modes produce different outputs on the same query."""
    torch.manual_seed(42)
    q = torch.randn(128)
    outs: dict[str, torch.Tensor] = {}
    for mode in ("hard_topk", "softmax_full", "softmax_topk"):
        # Build banks with identical key/value init (force same seed).
        torch.manual_seed(42)
        cfg = _cfg(mode)
        bank = RegimeAxisRetrieval(cfg)
        bank.eval()
        outs[mode] = bank(q)
    # hard_topk has shape (32, 128); softmax_full has (64, 128); softmax_topk (32, 128)
    # The two (32, 128) outputs should differ since one is unweighted, other is weighted.
    assert outs["hard_topk"].shape == (32, 128)
    assert outs["softmax_full"].shape == (64, 128)
    assert outs["softmax_topk"].shape == (32, 128)
    # softmax_topk is weight-multiplied; magnitudes should be smaller than hard_topk
    # (since weights sum to 1 over 64, top-32 weights mean ~0.03 each).
    assert outs["softmax_topk"].abs().mean() < outs["hard_topk"].abs().mean()
