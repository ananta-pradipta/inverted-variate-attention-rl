"""Unit tests for the Robust-InVAR-RL Phase 1 group-DRO + top-bottom loss.

Test contract per the Phase 1 spec (2026-05-26):
1. q concentrates on the max-loss group after several DRO steps.
2. top-bottom loss has zero gradient on the middle (N - 2M) positions.
3. eta=0 reduces to plain ERM (weighted_loss == mean of per-group losses
   under a uniform initial q).
4. Singleton groups in a synthetic per-day batch do not produce NaN.
5. q always sums to 1 after each step.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.robust_invar_rl.group_dro_loss import (
    compute_top_bottom_loss,
    group_dro_step,
)


def _uniform_q(g: int, device: torch.device) -> torch.Tensor:
    return torch.full((g,), 1.0 / g, device=device, dtype=torch.float32)


def test_dro_weights_concentrate_on_max_loss_group():
    """After several steps with the same dominant loss in group 0, q[0] > 1/G."""
    g = 3
    device = torch.device("cpu")
    q = _uniform_q(g, device)
    # Persistent loss vector with group 0 dominant.
    per_group = torch.tensor([2.0, 0.5, 0.5], device=device, requires_grad=False)
    for _ in range(20):
        _, q = group_dro_step(per_group, q, eta=0.5)
    assert q[0].item() > 1.0 / g + 1e-3, (
        f"q[0]={q[0].item():.4f} did not concentrate above uniform 1/G={1/g:.4f}"
    )
    # Other groups must have lost mass relative to uniform.
    assert q[1].item() < 1.0 / g
    assert q[2].item() < 1.0 / g


def test_top_bottom_loss_ignores_middle():
    """With M=25, N=500, gradients on the middle 450 score positions are zero."""
    torch.manual_seed(0)
    n, m = 500, 25
    scores = torch.randn(n, requires_grad=True)
    returns = torch.randn(n)
    mask = torch.ones(n, dtype=torch.float32)
    loss = compute_top_bottom_loss(scores, returns, mask, M=m)
    loss.backward()
    grad = scores.grad
    assert grad is not None
    # Determine top-M and bottom-M positions from the original score tensor.
    with torch.no_grad():
        top_idx = torch.topk(scores, k=m, largest=True).indices
        bot_idx = torch.topk(scores, k=m, largest=False).indices
    sel = torch.zeros(n, dtype=torch.bool)
    sel[top_idx] = True
    sel[bot_idx] = True
    middle_grad = grad[~sel]
    assert torch.all(middle_grad == 0.0), (
        f"middle positions received nonzero gradient: "
        f"max|grad_middle|={middle_grad.abs().max().item():.6f}"
    )
    # Sanity: top+bottom positions should have at least some nonzero gradient.
    tail_grad = grad[sel]
    assert tail_grad.abs().sum().item() > 0.0, (
        "top/bottom positions received zero gradient; loss is degenerate"
    )


def test_canonical_preserved_when_eta_zero():
    """eta=0 reduces to weighted_loss = q_old @ per_group_losses (uniform = mean)."""
    g = 4
    device = torch.device("cpu")
    q = _uniform_q(g, device)
    per_group = torch.tensor([0.7, 0.1, 0.3, 0.5], device=device)
    weighted, q_new = group_dro_step(per_group, q, eta=0.0)
    expected = per_group.mean()
    assert torch.allclose(weighted, expected, atol=1e-7), (
        f"weighted={weighted.item():.6f} != mean={expected.item():.6f}"
    )
    assert torch.allclose(q_new, q, atol=1e-7), "q must be unchanged when eta=0"


def test_group_split_handles_singleton_groups():
    """A singleton stock (mask sum = 1) does not produce NaN in top-bottom."""
    n, m = 10, 5
    scores = torch.randn(n, requires_grad=True)
    returns = torch.randn(n)
    mask = torch.zeros(n, dtype=torch.float32)
    mask[3] = 1.0  # only one active stock
    loss = compute_top_bottom_loss(scores, returns, mask, M=m)
    assert torch.isfinite(loss).item(), f"loss is non-finite for singleton: {loss}"
    # Scalar zero (no gradient) is acceptable since 2*M > active count.
    assert loss.item() == 0.0, "expected zero loss when fewer than 2 active stocks"

    # Also test the group_dro_step path with a single-group EMA.
    q = _uniform_q(2, scores.device)
    per_group = torch.tensor([0.5, 0.0])
    weighted, q_new = group_dro_step(per_group, q, eta=0.1)
    assert torch.isfinite(weighted).item()
    assert torch.isfinite(q_new).all().item()


def test_q_state_normalizes_to_unit_sum():
    """q sums to 1 after each step across a range of eta and loss vectors."""
    rng = np.random.default_rng(42)
    g = 8
    q = _uniform_q(g, torch.device("cpu"))
    for _ in range(30):
        per_group = torch.tensor(
            rng.normal(loc=0.5, scale=0.3, size=g).clip(min=0.0),
            dtype=torch.float32,
        )
        eta = float(rng.uniform(0.0, 0.5))
        _, q = group_dro_step(per_group, q, eta=eta)
        s = float(q.sum().item())
        assert abs(s - 1.0) < 1e-6, f"q did not sum to 1: sum={s:.8f}, eta={eta}"
        assert (q >= 0.0).all().item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
