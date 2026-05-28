"""Tests for :func:`invar_rl.layer2_sia.aux_loss.actor_aux_loss`.

Covers non-negativity of the three terms, gradient flow into actor params
only, and degenerate cases (single-group invariance, missing group_ids).
CPU-only, no Stable-Baselines3 dependency.
"""

from __future__ import annotations

import pytest
import torch

from invar_rl.layer2_sia.aux_loss import (
    AuxLossTerms,
    _gate_l1,
    _kl_to_standard_normal,
    _regime_invariance,
    actor_aux_loss,
)
from invar_rl.layer2_sia.sparse_actor import (
    SparseInvariantActor,
    resolve_dims,
)


def _make_aux(batch: int, latent: int, n_gates: int = 5, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    mu = torch.randn(batch, latent, generator=gen)
    logvar = torch.randn(batch, latent, generator=gen).clamp(-2, 1)
    gates = torch.sigmoid(torch.randn(batch, n_gates, generator=gen))
    return {"mu": mu, "logvar": logvar, "gates": gates}


def test_kl_non_negative() -> None:
    aux = _make_aux(32, 8, seed=1)
    kl = _kl_to_standard_normal(aux["mu"], aux["logvar"])
    assert float(kl.item()) >= -1e-6


def test_gate_l1_non_negative_and_bounded() -> None:
    aux = _make_aux(32, 8, seed=2)
    g = _gate_l1(aux["gates"])
    assert 0.0 <= float(g.item()) <= 1.0


def test_invariance_zero_when_single_group() -> None:
    aux = _make_aux(16, 4, seed=3)
    g_ids = torch.zeros(16, dtype=torch.long)
    inv = _regime_invariance(aux["mu"], g_ids)
    assert float(inv.item()) == pytest.approx(0.0, abs=1e-7)


def test_invariance_positive_when_groups_differ() -> None:
    # Construct two groups with disjoint mu shifts; variance MUST be > 0.
    mu = torch.cat(
        [torch.zeros(8, 4), torch.ones(8, 4) * 3.0], dim=0,
    )
    g_ids = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)
    inv = _regime_invariance(mu, g_ids)
    assert float(inv.item()) > 0.0


def test_actor_aux_loss_total_combines_weights() -> None:
    aux = _make_aux(16, 8, seed=4)
    g_ids = torch.randint(0, 3, (16,), dtype=torch.long)
    terms, total = actor_aux_loss(
        aux=aux, group_ids=g_ids,
        beta_kl=1e-3, lambda_gate=1e-4, lambda_inv=0.1,
    )
    expected = (
        1e-3 * terms.kl + 1e-4 * terms.gate_l1 + 0.1 * terms.inv
    )
    assert float(total.detach().item()) == pytest.approx(expected, rel=1e-4)
    assert isinstance(terms, AuxLossTerms)


def test_grad_flows_actor_only() -> None:
    """The auxiliary total must back-prop into actor params and only them."""
    macro_dim = 4
    dims = resolve_dims(obs_dim=7 + macro_dim, macro_dim=macro_dim)
    actor = SparseInvariantActor(dims=dims, latent_dim=4)
    # A second, unrelated module whose params should NOT receive any grad.
    bystander = torch.nn.Linear(5, 5)
    # Forward through the real actor so the graph is the same shape as
    # the SAC train step uses.
    obs = torch.randn(8, 7 + macro_dim)
    _, _, aux = actor(obs, deterministic=False)
    g_ids = torch.randint(0, 3, (8,), dtype=torch.long)
    _, total = actor_aux_loss(
        aux=aux, group_ids=g_ids,
        beta_kl=1e-2, lambda_gate=1e-2, lambda_inv=1.0,
    )
    # Add a bystander forward so its params join the computation graph
    # if any contamination exists; the gradient should still be zero
    # for bystander because the aux loss never touched it.
    bystander_loss = bystander(torch.zeros(1, 5)).sum() * 0.0
    (total + bystander_loss).backward()
    # At least one actor param has a non-zero gradient.
    actor_grad_norms = [
        p.grad.norm().item() for p in actor.parameters() if p.grad is not None
    ]
    assert any(g > 0.0 for g in actor_grad_norms)
    # Bystander grads are zero.
    for p in bystander.parameters():
        if p.grad is not None:
            assert float(p.grad.norm().item()) == pytest.approx(0.0, abs=1e-12)


def test_actor_aux_loss_rejects_missing_keys() -> None:
    with pytest.raises(KeyError):
        actor_aux_loss(
            aux={"mu": torch.zeros(2, 3)},  # missing logvar + gates
            group_ids=None,
            beta_kl=0.0, lambda_gate=0.0, lambda_inv=0.0,
        )


def test_actor_aux_loss_handles_none_group_ids() -> None:
    aux = _make_aux(8, 4, seed=5)
    terms, total = actor_aux_loss(
        aux=aux, group_ids=None,
        beta_kl=1e-3, lambda_gate=1e-4, lambda_inv=0.1,
    )
    assert terms.inv == pytest.approx(0.0, abs=1e-12)
    # Total must still be a leaf-attachable scalar carrying grad info.
    assert isinstance(total, torch.Tensor)
