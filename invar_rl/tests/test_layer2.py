"""Tests for Layer 2, the differentiable allocation layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from invar_rl.common.config import load_layer2_config
from invar_rl.common.seeding import make_rng
from invar_rl.layer2_alloc.covariance import (
    estimate_covariance,
    factor_model_covariance,
    ledoit_wolf_constant_correlation,
)
from invar_rl.layer2_alloc.qp_layer import MeanVarianceQP
from invar_rl.layer2_alloc.topk_layer import SoftTopKLongShort

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
N = 30


def _cfg():
    return load_layer2_config(CONFIG_DIR / "layer2.yaml")


def _synthetic_returns(seed: int = 42, t: int = 240, n: int = N) -> np.ndarray:
    rng = make_rng(seed)
    factors = rng.normal(0.0, 1.0, size=(t, 3))
    loadings = rng.normal(0.0, 1.0, size=(3, n))
    return factors @ loadings + rng.normal(0.0, 0.5, size=(t, n))


def _well_conditioned(cov: np.ndarray) -> bool:
    eig = np.linalg.eigvalsh(cov)
    return bool(
        np.allclose(cov, cov.T)
        and eig.min() > 0
        and np.isfinite(np.linalg.cond(cov))
    )


def test_covariance_estimators_are_well_conditioned() -> None:
    r = _synthetic_returns()
    lw = ledoit_wolf_constant_correlation(r)
    fm = factor_model_covariance(r, n_factors=3)
    assert lw.shape == (N, N)
    assert fm.shape == (N, N)
    assert _well_conditioned(lw)
    assert _well_conditioned(fm)


def _qp_inputs():
    r = _synthetic_returns()
    sigma = torch.from_numpy(
        estimate_covariance(r, "ledoit_wolf", 3)
    ).double()
    scores = torch.from_numpy(
        make_rng(7).normal(0.0, 1.0, size=N)
    ).double()
    return scores, sigma


def test_qp_produces_valid_long_short_book() -> None:
    cfg = _cfg()
    qp = MeanVarianceQP(cfg)
    scores, sigma = _qp_inputs()
    w, summary = qp(scores, sigma)
    # First-order conic solver tolerance: constraints hold to ~1e-3, which
    # is economically immaterial on a 0.05 per-name cap.
    tol = 1e-3
    assert torch.allclose(w.sum(), torch.zeros(()).double(), atol=tol)
    assert w.abs().sum().item() <= cfg.gross_leverage + tol
    assert w.abs().max().item() <= cfg.per_name_bound + tol
    assert set(summary) == {
        "predicted_vol",
        "effective_positions",
        "gross_exposure",
        "net_exposure",
    }


def test_topk_produces_valid_long_short_book() -> None:
    cfg = _cfg()
    topk = SoftTopKLongShort(cfg)
    scores, sigma = _qp_inputs()
    w, summary = topk(scores, sigma)
    assert torch.allclose(w.sum(), torch.zeros(()).double(), atol=1e-5)
    assert w.abs().sum().item() <= cfg.gross_leverage + 1e-4
    assert set(summary) == {
        "predicted_vol",
        "effective_positions",
        "gross_exposure",
        "net_exposure",
    }


def test_both_layers_share_summary_interface() -> None:
    cfg = _cfg()
    scores, sigma = _qp_inputs()
    _, qp_s = MeanVarianceQP(cfg)(scores, sigma)
    _, tk_s = SoftTopKLongShort(cfg)(scores, sigma)
    assert set(qp_s) == set(tk_s)
    for key in qp_s:
        assert qp_s[key].shape == tk_s[key].shape == torch.Size([])


def test_gradients_flow_to_scores_both_layers() -> None:
    cfg = _cfg()
    _, sigma = _qp_inputs()
    for layer in (MeanVarianceQP(cfg), SoftTopKLongShort(cfg)):
        s = torch.from_numpy(
            make_rng(7).normal(0.0, 1.0, size=N)
        ).double()
        s.requires_grad_(True)
        w, _ = layer(s, sigma)
        w.pow(2).sum().backward()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all()
        assert s.grad.abs().sum() > 0


def test_qp_gradient_matches_finite_difference() -> None:
    cfg = _cfg()
    qp = MeanVarianceQP(cfg)
    _, sigma = _qp_inputs()
    s = torch.from_numpy(
        make_rng(3).normal(0.0, 1.0, size=N)
    ).double()
    s.requires_grad_(True)

    def loss_of(scores: torch.Tensor) -> torch.Tensor:
        w, _ = qp(scores, sigma)
        return 0.5 * w.pow(2).sum()

    analytic = torch.autograd.grad(loss_of(s), s)[0]

    # A central-difference step large enough to clear the first-order
    # solver's residual noise floor. Correctness is judged by relative L2
    # error of the full gradient vector, which is robust to per-coordinate
    # solver jitter while still failing a genuinely wrong gradient.
    eps = 1e-4
    numeric = torch.zeros_like(s)
    with torch.no_grad():
        for i in range(N):
            d = torch.zeros_like(s)
            d[i] = eps
            numeric[i] = (loss_of(s + d) - loss_of(s - d)) / (2 * eps)
    rel_err = torch.linalg.norm(analytic - numeric) / (
        torch.linalg.norm(numeric) + 1e-12
    )
    assert torch.isfinite(analytic).all()
    assert rel_err.item() < 5e-2, f"relative gradient error {rel_err.item()}"
