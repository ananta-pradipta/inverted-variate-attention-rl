"""Phase 0 acceptance tests for RegimeXer-iT.

Six tests (the spec's seven minus `test_ccc_loss_at_optimum`, which is
dropped per the 2026-05-11 update that swaps CCC for the existing
hybrid_loss).
"""
from __future__ import annotations

import torch

from src.invar.baselines.regimexer import (
    RegimeXerIT,
    RegimeXerITConfig,
    count_parameters,
)
from src.invar.baselines.regimexer_blocks import FiLMBlock


def _make_inputs(N: int = 16, F: int = 26, F_m: int = 24, L: int = 60,
                 seed: int = 0):
    torch.manual_seed(seed)
    features = torch.randn(N, L, F)
    macro = torch.randn(L, F_m)
    mask = torch.ones(N, dtype=torch.bool)
    return features, macro, mask


def test_film_forward_shape() -> None:
    film = FiLMBlock(d_model=128, n_panel=26)
    N, F, d = 16, 26, 128
    H = torch.randn(N, F, d)
    c = torch.randn(N, d)
    out = film(H, c)
    assert out.shape == (N, F, d)


def test_invariance_gate_zero_collapse() -> None:
    """With alpha forced to 0, model output equals the invariant pathway
    (panel-only, single iT block, mean over F, linear head)."""
    torch.manual_seed(0)
    model = RegimeXerIT(RegimeXerITConfig()).eval()
    features, macro, mask = _make_inputs()
    with torch.no_grad():
        out = model(features, macro, mask, force_alpha=0.0)
        # Recompute the invariant pathway by hand using the same weights.
        H0_stock = model.embed_stock(features.transpose(-1, -2))
        H3_base = model._invariant_pathway(H0_stock)
        z_base = H3_base.mean(dim=1)
        y_base = model.y_head(z_base).squeeze(-1) * mask.float()
    delta = (out["y_hat"] - y_base).abs().max().item()
    assert delta < 1.0e-5, (
        f"alpha=0 collapse delta {delta} exceeds 1e-5; the invariant "
        f"pathway is not being preserved when the gate is closed."
    )


def test_invariance_gate_one_open() -> None:
    """With alpha forced to 1, the model uses the macro-conditioned pathway
    exclusively, so changes in macro must propagate to y_hat."""
    torch.manual_seed(1)
    model = RegimeXerIT(RegimeXerITConfig()).eval()
    features, macro_a, mask = _make_inputs(seed=1)
    macro_b = macro_a + 0.5 * torch.randn_like(macro_a)
    with torch.no_grad():
        y_a = model(features, macro_a, mask, force_alpha=1.0)["y_hat"]
        y_b = model(features, macro_b, mask, force_alpha=1.0)["y_hat"]
    delta = (y_a - y_b).abs().max().item()
    assert delta > 1.0e-5, (
        f"alpha=1 macro perturbation did not change y_hat (delta {delta}); "
        f"macro is not propagating through the gated pathway."
    )


def test_macro_propagation() -> None:
    """At alpha = 1 a macro perturbation changes y_hat; at alpha = 0 it must
    not (within numerical tolerance)."""
    torch.manual_seed(2)
    model = RegimeXerIT(RegimeXerITConfig()).eval()
    features, macro_a, mask = _make_inputs(seed=2)
    macro_b = macro_a + 1.0 * torch.randn_like(macro_a)
    with torch.no_grad():
        y_a0 = model(features, macro_a, mask, force_alpha=0.0)["y_hat"]
        y_b0 = model(features, macro_b, mask, force_alpha=0.0)["y_hat"]
        y_a1 = model(features, macro_a, mask, force_alpha=1.0)["y_hat"]
        y_b1 = model(features, macro_b, mask, force_alpha=1.0)["y_hat"]
    delta_alpha0 = (y_a0 - y_b0).abs().max().item()
    delta_alpha1 = (y_a1 - y_b1).abs().max().item()
    assert delta_alpha0 < 1.0e-5, (
        f"macro propagated to y_hat at alpha=0 (delta {delta_alpha0}); "
        f"the invariant pathway must not see macro."
    )
    assert delta_alpha1 > delta_alpha0 + 1.0e-4, (
        f"alpha=1 delta {delta_alpha1} is not meaningfully larger than "
        f"alpha=0 delta {delta_alpha0}; macro is not propagating at alpha=1."
    )


def test_param_count() -> None:
    """RegimeXer-iT full mode has at most 1.2 * iTransformer baseline params.

    iTransformer baseline: 1,001,525 params at d=128, n_heads=4, n_layers=4,
    ffn_hidden=512. RegimeXer-iT uses d=128, n_heads=8, n_layers=3 +
    1-block thin twin, ffn_hidden=512. The bar is 1.2 * 1,001,525 =
    1,201,830 params.
    """
    model = RegimeXerIT(RegimeXerITConfig())
    n_params = count_parameters(model)
    print(f"RegimeXer-iT total trainable parameters: {n_params:,}")
    BAR = int(1.2 * 1_001_525)
    assert n_params <= BAR, (
        f"RegimeXer-iT param count {n_params:,} exceeds 1.2x iTransformer "
        f"baseline bar ({BAR:,}). Reduce capacity."
    )


def test_deterministic_seed_42() -> None:
    """Two consecutive forwards with seed 42 produce identical outputs."""
    torch.manual_seed(42)
    features, macro, mask = _make_inputs(seed=42)
    cfg = RegimeXerITConfig()
    torch.manual_seed(42)
    model_a = RegimeXerIT(cfg).eval()
    torch.manual_seed(42)
    model_b = RegimeXerIT(cfg).eval()
    with torch.no_grad():
        y_a = model_a(features, macro, mask, force_alpha=None)["y_hat"]
        y_b = model_b(features, macro, mask, force_alpha=None)["y_hat"]
    delta = (y_a - y_b).abs().max().item()
    assert delta < 1.0e-6, (
        f"two seeded forwards diverge by {delta}; nondeterminism somewhere."
    )


def test_gradient_flows_through_new_params() -> None:
    """All new parameters (FiLM gamma/beta, gate MLP, thin block, y_head, v_head)
    receive non-zero gradients on a random loss."""
    torch.manual_seed(3)
    model = RegimeXerIT(RegimeXerITConfig()).train()
    features, macro, mask = _make_inputs(seed=3)
    out = model(features, macro, mask)
    target = torch.randn_like(out["y_hat"])
    vol_target = torch.randn_like(out["vol_hat"])
    loss = (
        ((out["y_hat"] - target) ** 2).mean()
        + 0.1 * ((out["vol_hat"] - vol_target) ** 2).mean()
        + 0.001 * out["alpha"].mean()
    )
    loss.backward()
    new_modules = [
        ("film.mlp_gamma", model.film.mlp_gamma),
        ("film.mlp_beta", model.film.mlp_beta),
        ("gate.mlp[0]", model.gate.mlp[0]),
        ("gate.mlp[2]", model.gate.mlp[2]),
        ("block_thin.attn.in_proj", model.block_thin.attn.in_proj_weight),
        ("y_head", model.y_head),
        ("v_head", model.v_head),
    ]
    for name, mod in new_modules:
        if isinstance(mod, torch.Tensor):
            # in_proj_weight is a Tensor, not a Module.
            params = [mod]
        else:
            params = list(mod.parameters())
        for p in params:
            assert p.grad is not None, f"{name}: no gradient"
            assert p.grad.abs().sum().item() > 0.0, (
                f"{name}: all-zero gradient (sum |grad| = 0)"
            )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
