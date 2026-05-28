"""Tests for :class:`invar_rl.layer2_sia.sparse_actor.SparseInvariantActor`.

Covers obs slicing, forward shape, gate range, exposure range, mu/logvar
shapes, and the MLP helper. CPU-only, no Stable-Baselines3 dependency.
"""

from __future__ import annotations

import pytest
import torch

from invar_rl.layer2_sia.sparse_actor import (
    MLP,
    SIADims,
    SparseInvariantActor,
    _split_obs,
    resolve_dims,
)


def _make_obs(
    batch_size: int, macro_dim: int, l1_uncertainty: int = 0, seed: int = 0
) -> torch.Tensor:
    total = 7 + macro_dim + l1_uncertainty
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, total, generator=gen)


def test_resolve_dims_validates_layout() -> None:
    dims = resolve_dims(obs_dim=23, macro_dim=16, l1_uncertainty=0)
    assert dims.total == 23
    assert dims.macro == 16
    assert dims.l1_uncertainty == 0
    assert dims.macro_small_dim == 16
    with pytest.raises(ValueError):
        resolve_dims(obs_dim=24, macro_dim=16, l1_uncertainty=0)
    with pytest.raises(ValueError):
        resolve_dims(obs_dim=20, macro_dim=12, l1_uncertainty=0, macro_small_dim=0)


def test_split_obs_layout_matches_observation_module() -> None:
    """Slicing must match invar_rl.layer3_control.observation.build_observation.

    Layout: [disp(1), wrapper(2), risk(4), macro(M), opt l1u(1)].
    """
    macro_dim = 5
    obs = _make_obs(4, macro_dim)
    dims = resolve_dims(obs_dim=obs.shape[1], macro_dim=macro_dim)
    blocks = _split_obs(obs, dims)
    assert blocks["dispersion"].shape == (4, 1)
    assert blocks["wrapper_stats"].shape == (4, 2)
    assert blocks["risk_state"].shape == (4, 4)
    assert blocks["macro"].shape == (4, macro_dim)
    assert blocks["l1_uncertainty"].shape == (4, 1)
    assert torch.allclose(blocks["dispersion"], obs[:, 0:1])
    assert torch.allclose(blocks["wrapper_stats"], obs[:, 1:3])
    assert torch.allclose(blocks["risk_state"], obs[:, 3:7])
    assert torch.allclose(blocks["macro"], obs[:, 7:7 + macro_dim])
    # Without an l1u tail, the helper zero-pads to width 1.
    assert torch.all(blocks["l1_uncertainty"] == 0.0)


def test_split_obs_with_l1u_tail() -> None:
    macro_dim = 3
    obs = _make_obs(2, macro_dim, l1_uncertainty=1)
    dims = resolve_dims(
        obs_dim=obs.shape[1], macro_dim=macro_dim, l1_uncertainty=1,
    )
    blocks = _split_obs(obs, dims)
    assert torch.allclose(
        blocks["l1_uncertainty"], obs[:, 7 + macro_dim:7 + macro_dim + 1]
    )


def test_forward_shapes_and_ranges() -> None:
    macro_dim = 8
    dims = resolve_dims(obs_dim=7 + macro_dim, macro_dim=macro_dim)
    actor = SparseInvariantActor(
        dims=dims, latent_dim=16, actor_hidden=(64, 64),
    )
    obs = _make_obs(7, macro_dim)
    action_squashed, log_prob, aux = actor(obs, deterministic=False)
    # Action is tanh-squashed in [-1, 1].
    assert action_squashed.shape == (7, 1)
    assert torch.all(action_squashed >= -1.0 - 1e-6)
    assert torch.all(action_squashed <= 1.0 + 1e-6)
    # Log_prob is (B,) and finite.
    assert log_prob.shape == (7,)
    assert torch.all(torch.isfinite(log_prob))
    assert aux["mu"].shape == (7, 16)
    assert aux["logvar"].shape == (7, 16)
    assert aux["gates"].shape == (7, 5)
    assert torch.all(aux["gates"] >= 0.0)
    assert torch.all(aux["gates"] <= 1.0)


def test_deterministic_forward_is_reproducible() -> None:
    macro_dim = 4
    dims = resolve_dims(obs_dim=7 + macro_dim, macro_dim=macro_dim)
    actor = SparseInvariantActor(dims=dims, latent_dim=8)
    obs = _make_obs(5, macro_dim, seed=11)
    a1, _, _ = actor(obs, deterministic=True)
    a2, _, _ = actor(obs, deterministic=True)
    assert torch.allclose(a1, a2)


def test_stochastic_forward_varies() -> None:
    macro_dim = 4
    dims = resolve_dims(obs_dim=7 + macro_dim, macro_dim=macro_dim)
    actor = SparseInvariantActor(dims=dims, latent_dim=8)
    obs = _make_obs(64, macro_dim, seed=12)
    # Two stochastic passes should differ; very low probability of exact match.
    a1, _, _ = actor(obs, deterministic=False)
    a2, _, _ = actor(obs, deterministic=False)
    assert not torch.allclose(a1, a2)


def test_logvar_clamped() -> None:
    """Encode logvar is clamped inside the actor.

    The latent logvar is clamped to [2*_LOG_STD_MIN, 2*_LOG_STD_MAX] =
    [-10, 4]; verify the clamp is at least active and never explodes.
    """
    macro_dim = 3
    dims = resolve_dims(obs_dim=7 + macro_dim, macro_dim=macro_dim)
    actor = SparseInvariantActor(dims=dims, latent_dim=4)
    big_obs = 100.0 * _make_obs(8, macro_dim, seed=13)
    _, _, _, _ = actor.encode(big_obs)  # smoke; ensure no crash
    _, _, _gates, _ = actor.encode(big_obs)
    _, logvar, _, _ = actor.encode(big_obs)
    # Latent logvar is clamped to [-10, 4].
    assert torch.all(logvar >= -10.0 - 1e-6)
    assert torch.all(logvar <= 4.0 + 1e-6)


def test_mlp_layer_count_matches_hidden_spec() -> None:
    mlp = MLP(in_dim=3, hidden=(8, 16, 32), out_dim=1)
    n_modules = sum(1 for _ in mlp.net)
    assert n_modules == 7  # 4 Linear + 3 ReLU


def test_actor_rejects_bad_input() -> None:
    macro_dim = 4
    dims = resolve_dims(obs_dim=7 + macro_dim, macro_dim=macro_dim)
    actor = SparseInvariantActor(dims=dims, latent_dim=4)
    with pytest.raises(ValueError):
        actor(torch.zeros(11))  # 1-D
    with pytest.raises(ValueError):
        actor(torch.zeros(2, 12))  # wrong total dim (expected 11)
