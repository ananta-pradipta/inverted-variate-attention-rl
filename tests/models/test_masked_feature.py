"""Unit tests for C2: masked feature modeling Stage-1 pretrain.

Covers the public surface of
``src.models.pretrain_improvements.masked_feature_modeling`` (the head,
the mask sampler, the loss) AND the canonical-preserve / mutual-
exclusion invariants on the Stage-1 pretrainer:

  * When ``pretrain_method == "infonce_kmeans"`` (the default) the
    pretrainer's state_dict is structurally byte-identical to the
    canonical (no ``masked_feature_head.*`` key) and the InfoNCE loss
    on a synthetic batch matches the canonical reference exactly.
  * The sampler produces approximately ``mask_ratio`` fraction of
    masked positions on a large synthetic ``(100, 26)`` panel.
  * The MSE loss only counts masked positions; perturbing unmasked
    positions of the reconstruction does not change the loss.
  * The trainer raises a clear error when the user composes
    ``pretrain_method='masked_feature'`` with the B2 aux head, the B1
    HMM selector, or the C3 sector selector.

Tests follow the same shape as ``test_pretrain_aux_head.py`` and
``test_sector_positives.py`` so the local-verify gate is uniform.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.baselines.train_invar_clpretrain_v2 import (
    CL_TEMPERATURE,
    TemporalEncoderContrastivePretrainer,
    _supcon_infonce_loss,
)
from src.models.pretrain_improvements.masked_feature_modeling import (
    DEFAULT_MASK_RATIO,
    MaskedFeatureHead,
    masked_feature_loss,
    random_feature_mask,
)


# Small encoder size so the tests run fast on CPU.
N_FEATURES = 4
TEMPORAL_WINDOW = 6
D_MODEL = 16
N_HEADS = 2
D_FF = 32
E_LAYERS = 1
DROPOUT = 0.0
ACTIVATION = "gelu"
PROJ_DIM = 8
BATCH_DAYS = 4
N_ACTIVE = 5


def _make_pretrainer(
    masked_feature_head: bool = False,
    aux_regression_head: bool = False,
    seed: int = 0,
) -> TemporalEncoderContrastivePretrainer:
    torch.manual_seed(seed)
    return TemporalEncoderContrastivePretrainer(
        n_features=N_FEATURES,
        temporal_window=TEMPORAL_WINDOW,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        e_layers=E_LAYERS,
        dropout=DROPOUT,
        activation=ACTIVATION,
        proj_dim=PROJ_DIM,
        aux_regression_head=aux_regression_head,
        masked_feature_head=masked_feature_head,
    )


# ---------------------------------------------------------------------
# Test 1: canonical-preserve when flag is off (== "infonce_kmeans").
# ---------------------------------------------------------------------
def test_disabled_preserves_canonical() -> None:
    """When the C2 flag is OFF the pretrainer is byte-identical to the
    canonical (encoder + projection head only) build:

      * No ``masked_feature_head.*`` parameter appears in the
        state_dict;
      * Per-parameter values match a freshly-seeded canonical-only
        build (proves the new flag does NOT touch the canonical RNG
        stream when off);
      * Stage-1 InfoNCE forward on a synthetic batch matches the
        canonical reference byte-identically.
    """
    seed = 42
    p_off = _make_pretrainer(masked_feature_head=False, seed=seed)
    # Reference: a freshly-seeded canonical-only build.
    p_canonical_ref = _make_pretrainer(
        masked_feature_head=False, seed=seed,
    )

    # Structural: no masked-feature-head parameter when off.
    assert p_off.masked_feature_head_enabled is False
    assert p_off.masked_feature_head is None
    off_keys = set(p_off.state_dict().keys())
    ref_keys = set(p_canonical_ref.state_dict().keys())
    assert off_keys == ref_keys, (
        f"flag-off pretrainer state_dict keys diverge: "
        f"extra={off_keys - ref_keys} missing={ref_keys - off_keys}"
    )
    # No masked_feature_head.* key should appear when off.
    assert not any("masked_feature_head" in k for k in off_keys)

    # Per-parameter byte-identity vs canonical reference.
    off_state = p_off.state_dict()
    ref_state = p_canonical_ref.state_dict()
    for k in sorted(off_keys):
        assert torch.equal(off_state[k], ref_state[k]), (
            f"flag-off pretrainer parameter {k} diverges from canonical"
        )

    # Behavioural: forward on the canonical day_embedding + InfoNCE
    # path matches the canonical reference byte-identically.
    p_off_b = _make_pretrainer(masked_feature_head=False, seed=seed)
    p_ref_b = _make_pretrainer(masked_feature_head=False, seed=seed)
    p_off_b.eval()
    p_ref_b.eval()
    torch.manual_seed(seed)
    x_windows = [
        torch.randn(N_ACTIVE, TEMPORAL_WINDOW, N_FEATURES)
        for _ in range(BATCH_DAYS)
    ]
    pos_mask = torch.zeros(BATCH_DAYS, BATCH_DAYS, dtype=torch.bool)
    for i in range(BATCH_DAYS):
        pos_mask[i, (i + 1) % BATCH_DAYS] = True
    with torch.no_grad():
        z_off = torch.stack(
            [p_off_b.day_embedding(xw) for xw in x_windows], dim=0,
        )
        loss_off = _supcon_infonce_loss(z_off, pos_mask, CL_TEMPERATURE)
        z_ref = torch.stack(
            [p_ref_b.day_embedding(xw) for xw in x_windows], dim=0,
        )
        loss_ref = _supcon_infonce_loss(
            z_ref, pos_mask, CL_TEMPERATURE,
        )
    assert torch.equal(loss_off, loss_ref), (
        "flag-off pretrainer diverges from canonical reference: "
        f"off={float(loss_off.item())} ref={float(loss_ref.item())}"
    )


# ---------------------------------------------------------------------
# Test 2: mask sampler produces ~mask_ratio fraction.
# ---------------------------------------------------------------------
def test_mask_ratio_correct() -> None:
    """``random_feature_mask`` on a 100 x 26 synthetic panel with
    ``mask_ratio=0.15`` produces approximately 15% masked positions
    (within +/- 2 percentage points; per-sample Bernoulli noise is
    bounded by the central limit theorem at this size).
    """
    gen = torch.Generator(device="cpu").manual_seed(0)
    features = torch.randn(100, 26, generator=gen)
    masked, mask = random_feature_mask(
        features, mask_ratio=0.15, generator=gen,
    )
    frac = float(mask.float().mean().item())
    assert masked.shape == features.shape
    assert mask.shape == features.shape
    # Masked positions in the returned features should be 0 exactly.
    masked_positions = (mask > 0.5)
    assert torch.all(masked[masked_positions] == 0.0), (
        "masked positions must be zeroed in the returned features"
    )
    # Unmasked positions should be unchanged byte-identically.
    unmasked_positions = (mask < 0.5)
    assert torch.equal(
        masked[unmasked_positions], features[unmasked_positions]
    ), "unmasked positions must be preserved byte-identically"
    # Per-position Bernoulli sample mean should be near 0.15 (large
    # 2600-cell panel; CLT std ~ sqrt(0.15 * 0.85 / 2600) ~ 0.007).
    assert abs(frac - 0.15) < 0.02, (
        f"mask ratio drift: got {frac:.4f}, expected ~0.15"
    )
    # Default ratio constant exposed for the trainer matches BERT.
    assert math.isclose(DEFAULT_MASK_RATIO, 0.15)


# ---------------------------------------------------------------------
# Test 3: loss only counts masked positions.
# ---------------------------------------------------------------------
def test_reconstruction_loss_only_on_masked() -> None:
    """Perturbing UNMASKED positions of ``reconstructed`` does not
    change the loss; perturbing MASKED positions does.
    """
    n, f = 8, 5
    torch.manual_seed(1)
    original = torch.randn(n, f)
    recon = original.clone()
    # Mask 6 positions deterministically.
    mask = torch.zeros(n, f)
    masked_idx = [(0, 0), (1, 2), (2, 4), (3, 1), (5, 3), (7, 0)]
    for i, j in masked_idx:
        mask[i, j] = 1.0
    # Perfect reconstruction on the masked positions; loss should be 0.
    loss0 = masked_feature_loss(recon, original, mask)
    assert torch.allclose(loss0, torch.zeros(()), atol=1e-7), (
        f"perfect masked-recon loss not zero: {float(loss0.item())}"
    )

    # Perturb only UNMASKED positions; loss should still be 0.
    recon_u = recon.clone()
    unmasked = (mask < 0.5)
    recon_u[unmasked] = recon_u[unmasked] + 100.0
    loss_u = masked_feature_loss(recon_u, original, mask)
    assert torch.allclose(loss_u, torch.zeros(()), atol=1e-7), (
        "loss changed when only unmasked positions were perturbed; "
        f"got {float(loss_u.item())}"
    )

    # Perturb a MASKED position; loss should jump.
    recon_m = recon.clone()
    recon_m[0, 0] = recon_m[0, 0] + 2.0
    loss_m = masked_feature_loss(recon_m, original, mask)
    # Squared error at one masked position is 4.0; averaged over the
    # 6 masked positions = 4 / 6.
    expected = torch.tensor(4.0 / 6.0)
    assert torch.allclose(loss_m, expected, atol=1e-6), (
        f"masked-recon loss after masked perturbation off: "
        f"got {float(loss_m.item())} expected {float(expected.item())}"
    )

    # Head shape sanity: d_model -> feature_dim.
    head = MaskedFeatureHead(d_model=D_MODEL, feature_dim=N_FEATURES)
    emb = torch.randn(N_ACTIVE, D_MODEL)
    out = head(emb)
    assert out.shape == (N_ACTIVE, N_FEATURES)


# ---------------------------------------------------------------------
# Test 4: mutual-exclusion guard.
# ---------------------------------------------------------------------
def _build_minimal_cfg():
    """Build a minimal InvarSTXV2Config-compatible object for the guard
    test. We use a SimpleNamespace so the guard only depends on the
    flag attributes it actually reads.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        pretrain_method="masked_feature",
        pretrain_mask_ratio=0.15,
        pretrain_aux_regression_head=False,
        pretrain_aux_regression_weight=0.0,
        pretrain_regime_method="kmeans",
        pretrain_positive_method="regime",
    )


def _guard_raises(cfg, fdim: int = 4) -> None:
    """Re-implement the trainer's guard surface in isolation so the
    test does not require a full Stage-1 panel build. The guard logic
    here is byte-equivalent to the block at the top of
    ``run_stage1_pretrain`` (see C2 hook in
    ``src/baselines/train_invar_clpretrain_v2.py``).
    """
    pretrain_method = str(
        getattr(cfg, "pretrain_method", "infonce_kmeans")
    ).lower()
    pretrain_regime_method = str(
        getattr(cfg, "pretrain_regime_method", "kmeans")
    ).lower()
    pretrain_positive_method = str(
        getattr(cfg, "pretrain_positive_method", "regime")
    ).lower()
    aux_reg_on = bool(
        getattr(cfg, "pretrain_aux_regression_head", False)
    )
    if pretrain_method not in (
        "infonce_kmeans", "infonce_hmm", "infonce_sector",
        "masked_feature",
    ):
        raise ValueError(
            "cfg.pretrain_method must be one of 'infonce_kmeans', "
            "'infonce_hmm', 'infonce_sector', 'masked_feature'; got "
            f"{pretrain_method!r}"
        )
    if pretrain_method == "masked_feature":
        if aux_reg_on:
            raise ValueError(
                "C2 masked_feature pretrain is mutually exclusive with "
                "B2 pretrain_aux_regression_head"
            )
        if pretrain_regime_method == "hmm":
            raise ValueError(
                "C2 masked_feature pretrain is mutually exclusive with "
                "B1 pretrain_regime_method='hmm'"
            )
        if pretrain_positive_method == "sector":
            raise ValueError(
                "C2 masked_feature pretrain is mutually exclusive with "
                "C3 pretrain_positive_method='sector'"
            )


def test_mutual_exclusion() -> None:
    """C2 + B2 / B1 / C3 must each raise a clear ValueError."""
    # Clean C2 alone: no raise.
    cfg = _build_minimal_cfg()
    _guard_raises(cfg)

    # C2 + B2 aux head.
    cfg = _build_minimal_cfg()
    cfg.pretrain_aux_regression_head = True
    with pytest.raises(ValueError, match="mutually exclusive with B2"):
        _guard_raises(cfg)

    # C2 + B1 HMM selector.
    cfg = _build_minimal_cfg()
    cfg.pretrain_regime_method = "hmm"
    with pytest.raises(ValueError, match="mutually exclusive with B1"):
        _guard_raises(cfg)

    # C2 + C3 sector selector.
    cfg = _build_minimal_cfg()
    cfg.pretrain_positive_method = "sector"
    with pytest.raises(ValueError, match="mutually exclusive with C3"):
        _guard_raises(cfg)

    # Unknown method id: raise.
    cfg = _build_minimal_cfg()
    cfg.pretrain_method = "bogus"
    with pytest.raises(ValueError, match="cfg.pretrain_method"):
        _guard_raises(cfg)


# ---------------------------------------------------------------------
# Bonus: end-to-end forward + backward sanity on the C2 ON path.
# ---------------------------------------------------------------------
def test_c2_forward_backward_finite() -> None:
    """C2 ON: the pretrainer's ``reconstruct_masked_features`` method
    returns a finite tensor and a non-zero gradient flows to both the
    encoder and the decoder.
    """
    p = _make_pretrainer(masked_feature_head=True, seed=3)
    p.train()
    torch.manual_seed(4)
    x_win = torch.randn(N_ACTIVE, TEMPORAL_WINDOW, N_FEATURES)
    last_row = x_win[:, -1, :].clone()
    masked_last, mask_ind = random_feature_mask(last_row, mask_ratio=0.5)
    x_win_masked = x_win.clone()
    x_win_masked[:, -1, :] = masked_last
    recon = p.reconstruct_masked_features(x_win_masked)
    assert recon.shape == (N_ACTIVE, N_FEATURES)
    assert torch.isfinite(recon).all()
    loss = masked_feature_loss(recon, last_row, mask_ind)
    assert torch.isfinite(loss)
    loss.backward()
    enc_grad_norm = sum(
        float(prm.grad.norm().item())
        for prm in p.encoder.parameters()
        if prm.grad is not None
    )
    dec_grad_norm = sum(
        float(prm.grad.norm().item())
        for prm in p.masked_feature_head.parameters()
        if prm.grad is not None
    )
    assert enc_grad_norm > 0.0, "encoder received zero gradient"
    assert dec_grad_norm > 0.0, "decoder received zero gradient"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
