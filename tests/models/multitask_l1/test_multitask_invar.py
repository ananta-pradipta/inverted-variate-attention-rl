"""Unit tests for Option B's multi-task temporal encoder.

Synthetic 3-universe batch checks:
  * Forward pass produces ``(N_active, d_model)`` per universe.
  * Backward pass produces gradients on BOTH the universe-specific
    projection AND the shared backbone.
  * State-dict assembly produces keys that match the canonical
    ``PerTickerTemporalEncoder.state_dict()`` BYTE-FOR-BYTE, and the
    assembled state loads into a fresh canonical encoder with
    ``strict=True`` (no missing / no unexpected keys).
  * Cross-universe state-dict assembly differs ONLY in input_proj
    weights / bias and the shared keys are bitwise-identical across
    universes.

These checks gate the Wulver smoke; running ``pytest -q
tests/models/multitask_l1/test_multitask_invar.py`` locally is the
mandated "Local unit tests BEFORE Wulver smoke" step.
"""
from __future__ import annotations

import torch

from src.baselines.train_invar_stx_v2 import PerTickerTemporalEncoder
from src.models.multitask_l1 import (
    MultitaskTemporalEncoder,
    MultitaskTemporalEncoderConfig,
    UNIVERSE_FEATURE_DIMS,
    assemble_per_universe_encoder_state,
)
from src.models.multitask_l1.multitask_invar import (
    expected_canonical_encoder_keys,
)


def _toy_cfg() -> MultitaskTemporalEncoderConfig:
    # Small but realistic shapes; mirrors canonical defaults except for
    # the deliberately tiny e_layers / d_ff to keep the test fast.
    return MultitaskTemporalEncoderConfig(
        temporal_window=8,
        d_model=16,
        n_heads=2,
        d_ff=32,
        e_layers=1,
        dropout=0.0,
        activation="gelu",
        universe_feature_dims={
            "lattice_native": 26,
            "nasdaq100": 26,
            "biotech_nbi_enriched": 22,
        },
    )


def test_default_universe_dims_match_panels() -> None:
    """The module-level dict must match each panel module's len(FEATURE_COLS)."""
    assert UNIVERSE_FEATURE_DIMS == {
        "lattice_native": 26,
        "nasdaq100": 26,
        "biotech_nbi_enriched": 22,
    }


def test_forward_per_universe_shapes() -> None:
    """Forward must accept each registered universe's feature width."""
    cfg = _toy_cfg()
    enc = MultitaskTemporalEncoder(cfg)
    enc.eval()
    cases = [
        ("lattice_native", 5, 26),
        ("nasdaq100", 7, 26),
        ("biotech_nbi_enriched", 3, 22),
    ]
    for uid, n_active, fdim in cases:
        x = torch.randn(n_active, cfg.temporal_window, fdim)
        out = enc(x, uid)
        assert out.shape == (n_active, cfg.d_model), (
            f"universe={uid} expected (N={n_active}, d={cfg.d_model}); "
            f"got tuple({tuple(out.shape)})."
        )


def test_forward_wrong_feature_width_rejected() -> None:
    """Mismatched F_u must error explicitly, not silently broadcast."""
    cfg = _toy_cfg()
    enc = MultitaskTemporalEncoder(cfg)
    enc.eval()
    bad = torch.randn(3, cfg.temporal_window, 99)  # any non-22 / non-26
    raised = False
    try:
        enc(bad, "lattice_native")
    except ValueError:
        raised = True
    assert raised, "[ERR] expected ValueError on wrong-F input width"


def test_backward_produces_grads_on_shared_and_universe_params() -> None:
    """A synthetic 3-universe minibatch must train BOTH the shared
    backbone and every per-universe input projection."""
    torch.manual_seed(0)
    cfg = _toy_cfg()
    enc = MultitaskTemporalEncoder(cfg)
    enc.train()

    universes = ["lattice_native", "nasdaq100", "biotech_nbi_enriched"]
    n_active_per_u = {"lattice_native": 5, "nasdaq100": 7, "biotech_nbi_enriched": 3}

    optim = torch.optim.SGD(enc.parameters(), lr=1.0e-3)
    optim.zero_grad()
    total_loss = torch.zeros((), requires_grad=False)
    grad_sum = None
    for uid in universes:
        fdim = cfg.universe_feature_dims[uid]
        x = torch.randn(n_active_per_u[uid], cfg.temporal_window, fdim)
        out = enc(x, uid)
        # Trivial target: mean of all activations; backprop drives the
        # whole graph (per-universe proj + shared backbone).
        loss = out.pow(2).mean()
        total_loss = total_loss + loss.detach()
        loss.backward()
        grad_sum = float(loss.detach().item()) if grad_sum is None else grad_sum

    # Shared-backbone params must all have grads after the joint pass.
    for name, p in enc.shared_backbone.named_parameters():
        assert p.grad is not None, (
            f"[ERR] shared backbone param {name} did NOT receive a grad"
        )
        assert torch.isfinite(p.grad).all(), (
            f"[ERR] shared backbone param {name} grad is non-finite"
        )

    # Each universe's projection must have grads.
    for uid in universes:
        proj = enc.universe_input_projs[uid]
        for name, p in proj.named_parameters():
            assert p.grad is not None, (
                f"[ERR] universe={uid} proj.{name} did NOT receive a grad"
            )
            assert torch.isfinite(p.grad).all(), (
                f"[ERR] universe={uid} proj.{name} grad is non-finite"
            )

    optim.step()
    assert total_loss.item() > 0.0


def test_assembled_state_dict_matches_canonical_keys() -> None:
    """assemble_per_universe_encoder_state must produce the EXACT
    canonical key set so the Stage-2 strict-load works unchanged."""
    cfg = _toy_cfg()
    enc = MultitaskTemporalEncoder(cfg)
    for uid, fdim in cfg.universe_feature_dims.items():
        assembled = assemble_per_universe_encoder_state(enc, uid)
        expected_keys = expected_canonical_encoder_keys(fdim, cfg)
        got_keys = sorted(assembled.keys())
        assert got_keys == expected_keys, (
            f"[ERR] universe={uid} state-dict key mismatch\n"
            f"  missing:    {sorted(set(expected_keys) - set(got_keys))}\n"
            f"  unexpected: {sorted(set(got_keys) - set(expected_keys))}"
        )


def test_assembled_state_loads_into_canonical_encoder_strict() -> None:
    """Strict load into a fresh canonical PerTickerTemporalEncoder must
    succeed for every registered universe."""
    cfg = _toy_cfg()
    enc = MultitaskTemporalEncoder(cfg)
    enc.eval()
    for uid, fdim in cfg.universe_feature_dims.items():
        assembled = assemble_per_universe_encoder_state(enc, uid)
        canonical = PerTickerTemporalEncoder(
            n_features=fdim,
            temporal_window=cfg.temporal_window,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            e_layers=cfg.e_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )
        incompat = canonical.load_state_dict(assembled, strict=True)
        assert not incompat.missing_keys, (
            f"[ERR] universe={uid} strict load missing: {incompat.missing_keys}"
        )
        assert not incompat.unexpected_keys, (
            f"[ERR] universe={uid} strict load unexpected: {incompat.unexpected_keys}"
        )
        # Forward through the freshly loaded canonical encoder; output
        # must equal the multitask encoder's output (modulo dropout=0).
        x = torch.randn(4, cfg.temporal_window, fdim)
        with torch.no_grad():
            out_canon = canonical(x)
            out_multi = enc(x, uid)
        assert torch.allclose(out_canon, out_multi, atol=1.0e-6), (
            f"[ERR] universe={uid} forward mismatch between canonical "
            f"and multitask encoder after state-dict re-assembly"
        )


def test_cross_universe_assemblies_share_backbone_bitwise() -> None:
    """The shared backbone tensors must be byte-identical across the
    per-universe assembled state dicts (only input_proj differs)."""
    cfg = _toy_cfg()
    enc = MultitaskTemporalEncoder(cfg)
    universes = list(cfg.universe_feature_dims.keys())
    base = assemble_per_universe_encoder_state(enc, universes[0])
    for uid in universes[1:]:
        other = assemble_per_universe_encoder_state(enc, uid)
        for k in base:
            if k.startswith("input_proj"):
                # input_proj is universe-specific by design; should differ
                # (different shape OR different values).
                continue
            assert k in other, f"[ERR] universe={uid} missing shared key {k}"
            assert torch.equal(base[k], other[k]), (
                f"[ERR] shared backbone tensor {k} differs between "
                f"universes {universes[0]} and {uid}"
            )


def test_unknown_universe_id_rejected() -> None:
    cfg = _toy_cfg()
    enc = MultitaskTemporalEncoder(cfg)
    raised = False
    try:
        enc(torch.randn(2, cfg.temporal_window, 26), "no_such_universe")
    except KeyError:
        raised = True
    assert raised, "[ERR] expected KeyError on unknown universe id"
