"""Tests for the Phase 4 ``no_s`` ablation (SparseInvariantActor sparse_gates=False).

When ``sparse_gates=False`` the actor's per-block gates collapse to constant
1.0 for any input batch. The KL latent bottleneck (mu / logvar) and the rest
of the actor pipeline must still run unchanged; only the sparse routing is
disabled. CPU-only; no Stable-Baselines3 dependency.
"""

from __future__ import annotations

import torch

from invar_rl.layer2_sia.sparse_actor import (
    SparseInvariantActor,
    resolve_dims,
)


def _make_obs(
    batch_size: int, macro_dim: int, l1_uncertainty: int = 0, seed: int = 0
) -> torch.Tensor:
    total = 7 + macro_dim + l1_uncertainty
    gen = torch.Generator().manual_seed(int(seed))
    return torch.randn(batch_size, total, generator=gen)


def test_no_sparse_gates_gates_are_constant_one_for_any_input() -> None:
    """gates == ones((B, 5)) regardless of the input observation."""
    macro_dim = 8
    dims = resolve_dims(obs_dim=7 + macro_dim, macro_dim=macro_dim)
    actor = SparseInvariantActor(
        dims=dims, latent_dim=16, actor_hidden=(64, 64),
        sparse_gates=False,
    )
    # Two unrelated obs batches: the gates must be identical (and == 1.0).
    obs_a = _make_obs(11, macro_dim, seed=1)
    obs_b = _make_obs(11, macro_dim, seed=999)
    _, _, _, _ = actor.encode(obs_a)
    _, _, _, _ = actor.encode(obs_b)
    _, _, aux_a = actor(obs_a, deterministic=True)
    _, _, aux_b = actor(obs_b, deterministic=True)
    assert aux_a["gates"].shape == (11, 5)
    assert aux_b["gates"].shape == (11, 5)
    expected = torch.ones((11, 5), dtype=obs_a.dtype)
    assert torch.allclose(aux_a["gates"], expected)
    assert torch.allclose(aux_b["gates"], expected)


def test_no_sparse_gates_pipeline_still_returns_finite_action() -> None:
    """The rest of the actor (mu_net, logvar_net, action heads) still runs."""
    macro_dim = 5
    dims = resolve_dims(obs_dim=7 + macro_dim, macro_dim=macro_dim)
    actor = SparseInvariantActor(
        dims=dims, latent_dim=8, sparse_gates=False,
    )
    obs = _make_obs(9, macro_dim, seed=42)
    action_squashed, log_prob, aux = actor(obs, deterministic=False)
    assert action_squashed.shape == (9, 1)
    assert torch.all(action_squashed >= -1.0 - 1e-6)
    assert torch.all(action_squashed <= 1.0 + 1e-6)
    assert log_prob.shape == (9,)
    assert torch.all(torch.isfinite(log_prob))
    assert aux["mu"].shape == (9, 8)
    assert aux["logvar"].shape == (9, 8)


def test_no_sparse_gates_default_remains_sparse() -> None:
    """Construction without the flag preserves the sparse gating behaviour."""
    macro_dim = 4
    dims = resolve_dims(obs_dim=7 + macro_dim, macro_dim=macro_dim)
    actor_default = SparseInvariantActor(dims=dims, latent_dim=8)
    actor_no_s = SparseInvariantActor(
        dims=dims, latent_dim=8, sparse_gates=False,
    )
    obs = _make_obs(5, macro_dim, seed=7)
    _, _, aux_default = actor_default(obs, deterministic=True)
    _, _, aux_no_s = actor_no_s(obs, deterministic=True)
    # Default: gates can be anywhere in [0, 1] (non-trivial sigmoid).
    assert (aux_default["gates"] < 1.0 - 1e-6).any() or (
        aux_default["gates"] > 1e-6
    ).any()
    # no_s: gates pinned to exactly 1.0.
    assert torch.allclose(aux_no_s["gates"], torch.ones_like(aux_no_s["gates"]))
