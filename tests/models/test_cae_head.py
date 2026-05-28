"""Unit tests for src/models/resinvar/cae_head.py (ResInVAR-RL Phase 1)."""
from __future__ import annotations

import pytest
import torch

from src.models.resinvar import CAEHead
from src.models.resinvar.cae_head import _factor_regression
from src.utils.seed import seed_all


def _make_head(feature_dim: int = 26, k_latent: int = 3) -> CAEHead:
    seed_all(42)
    return CAEHead(feature_dim=feature_dim, k_latent=k_latent)


def test_forward_shapes() -> None:
    n, f, k = 20, 26, 3
    head = _make_head(feature_dim=f, k_latent=k)
    x = torch.randn(n, f)
    r = torch.randn(n)
    mask = torch.ones(n, dtype=torch.bool)

    out = head.forward(x, r, mask)

    assert out["B"].shape == (n, k)
    assert out["f"].shape == (k,)
    assert out["r_hat"].shape == (n,)
    assert out["eps"].shape == (n,)


def test_reconstruction_loss_nonneg() -> None:
    n, f, k = 16, 26, 3
    head = _make_head(feature_dim=f, k_latent=k)
    x = torch.randn(n, f)
    r = torch.randn(n)
    mask = torch.ones(n, dtype=torch.bool)

    out = head.forward(x, r, mask)
    loss = head.reconstruction_loss(out["eps"], mask)

    assert loss.dim() == 0
    assert float(loss.item()) >= 0.0


def test_factor_regression_closed_form() -> None:
    """Standalone closed-form helper recovers a known f within 1e-4.

    The MLP loading branch cannot recover an arbitrary B_true from
    arbitrary x_t, so we test the underlying _factor_regression helper
    directly. This is the recommended path in the Phase 1 spec.
    """
    seed_all(0)
    n, k = 64, 3
    # Build B with orthonormal columns via QR decomposition.
    rand_mat = torch.randn(n, k, dtype=torch.float64)
    b_true, _ = torch.linalg.qr(rand_mat)
    f_true = torch.tensor([0.7, -1.3, 2.1], dtype=torch.float64)
    r = b_true @ f_true

    # Tight bound: with ridge=0 and exact orthonormal B, recovery is exact
    # up to float64 numerics, well inside the spec's 1e-4 tolerance.
    f_hat = _factor_regression(b_true, r, ridge_lambda=0.0)
    err = (f_hat - f_true).abs().max().item()
    assert err < 1e-4, f"factor recovery err {err:.3e} exceeds 1e-4"
    # Also check the ridge=1e-3 default: bias is |f|_max * lambda/(1+lambda).
    f_hat_ridge = _factor_regression(b_true, r, ridge_lambda=1e-3)
    err_ridge = (f_hat_ridge - f_true).abs().max().item()
    assert err_ridge < 5e-3, (
        f"ridge=1e-3 recovery err {err_ridge:.3e} larger than expected bias"
    )


def test_orthogonality_penalty_zero_when_orthogonal() -> None:
    """Penalty is near zero for a centered, orthonormal-scaled B.

    The module recenters B internally before forming the Gram matrix,
    so we construct a zero-mean orthonormal-column B by QR-ing a
    centered random matrix, then rescaling each column to unit norm
    AFTER centering. We use float64 throughout to keep the residual
    Frobenius error inside the 1e-6 tolerance.
    """
    seed_all(1)
    n, k, f = 128, 3, 26
    head = CAEHead(feature_dim=f, k_latent=k)
    rand_mat = torch.randn(n, k, dtype=torch.float64)
    rand_centered = rand_mat - rand_mat.mean(dim=0, keepdim=True)
    q, _ = torch.linalg.qr(rand_centered)
    # q has orthonormal columns AND zero column means (centering is
    # preserved by QR when applied to a centered matrix because the
    # all-ones vector is in the orthogonal complement of q's columns).
    # B^T B / N == I requires scaling columns by sqrt(N).
    b = q * float(n) ** 0.5
    mask = torch.ones(n, dtype=torch.bool)

    pen = head.orthogonality_penalty(b, mask)

    assert float(pen.item()) < 1e-6, (
        f"orthogonality penalty {float(pen.item()):.3e} not near zero"
    )


def test_active_mask_respected() -> None:
    n, f, k = 20, 26, 3
    head = _make_head(feature_dim=f, k_latent=k)
    x = torch.randn(n, f, requires_grad=True)
    r = torch.randn(n)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:10] = True

    out = head.forward(x, r, mask)
    loss = head.reconstruction_loss(out["eps"], mask)
    loss.backward()

    assert x.grad is not None
    grad_inactive = x.grad[10:]
    assert torch.all(grad_inactive == 0.0), (
        "inactive stocks must not receive gradient"
    )
    # Sanity: at least some active rows should have nonzero gradient.
    assert torch.any(x.grad[:10] != 0.0), "active rows must have gradient"


def test_deterministic_seed() -> None:
    n, f, k = 24, 26, 3

    def _run() -> dict:
        seed_all(42)
        head = CAEHead(feature_dim=f, k_latent=k)
        x = torch.randn(n, f)
        r = torch.randn(n)
        mask = torch.ones(n, dtype=torch.bool)
        return head.forward(x, r, mask)

    out_a = _run()
    out_b = _run()

    for key in ("B", "f", "r_hat", "eps"):
        diff = (out_a[key] - out_b[key]).abs().max().item()
        assert diff < 1e-6, f"non-deterministic output for {key}: diff={diff:.3e}"


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_k_latent_variants(k: int) -> None:
    n, f = 18, 26
    seed_all(7)
    head = CAEHead(feature_dim=f, k_latent=k)
    x = torch.randn(n, f)
    r = torch.randn(n)
    mask = torch.ones(n, dtype=torch.bool)

    out = head.forward(x, r, mask)

    assert out["B"].shape == (n, k)
    assert out["f"].shape == (k,)
    assert out["r_hat"].shape == (n,)
    assert out["eps"].shape == (n,)
