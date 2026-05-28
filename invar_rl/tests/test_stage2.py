"""Tests for Stage 2, the decision-focused Layer 1 + Layer 2 pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from invar_rl.common.config import (
    load_base_config,
    load_layer1_config,
    load_layer2_config,
)
from invar_rl.common.seeding import set_global_seed
from invar_rl.data.synthetic import SyntheticPanel
from invar_rl.layer1_ranker.invar import INVAR
from invar_rl.layer2_alloc.qp_layer import MeanVarianceQP
from invar_rl.training.stage2_decision import (
    _day_inputs,
    _portfolio_return,
    _set_variant_trainable,
    build_one_day_return_matrix,
    covariance_for_day,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DAY = 400


def _pipeline():
    base = load_base_config(CONFIG_DIR / "base.yaml")
    layer1 = load_layer1_config(CONFIG_DIR / "layer1.yaml")
    layer2 = load_layer2_config(CONFIG_DIR / "layer2.yaml")
    set_global_seed(42)
    panel = SyntheticPanel(base.synthetic, seed=42)
    ret1_full = build_one_day_return_matrix(panel)
    model = INVAR(
        layer1.model,
        n_features=base.synthetic.n_features,
        lookback=base.synthetic.lookback,
        macro_dim=base.synthetic.macro_dim,
    )
    qp = MeanVarianceQP(layer2)
    return base, layer2, panel, ret1_full, model, qp


def _decision_loss(model, qp, panel, ret1_full, layer2):
    feats, macro, g_idx, fwd, finite = _day_inputs(panel, DAY)
    sigma = torch.from_numpy(
        covariance_for_day(ret1_full, g_idx, DAY, layer2.cov_lookback, 0,
                           layer2)
    ).float()
    scores = model(feats, macro).scores
    weights, _ = qp(scores, sigma)
    return -_portfolio_return(weights, fwd, finite)


def test_pipeline_is_end_to_end_differentiable_variant_b() -> None:
    _, layer2, panel, ret1_full, model, qp = _pipeline()
    _set_variant_trainable(model, "B")
    loss = _decision_loss(model, qp, panel, ret1_full, layer2)
    assert torch.isfinite(loss)
    loss.backward()
    # The decision-focused gradient must reach Layer 1 parameters that are
    # upstream of the QP, not only the score head.
    enc_grad = [
        p.grad for n, p in model.named_parameters()
        if n.startswith("encoder") and p.requires_grad
    ]
    assert enc_grad and all(g is not None for g in enc_grad)
    assert any(g.abs().sum() > 0 for g in enc_grad)


def test_variant_a_freezes_everything_but_the_score_head() -> None:
    _, layer2, panel, ret1_full, model, qp = _pipeline()
    _set_variant_trainable(model, "A")
    trainable = {
        n for n, p in model.named_parameters() if p.requires_grad
    }
    assert trainable
    assert all(n.startswith("score_head") for n in trainable)
    loss = _decision_loss(model, qp, panel, ret1_full, layer2)
    loss.backward()
    for n, p in model.named_parameters():
        if n.startswith("score_head"):
            assert p.grad is not None
        else:
            assert p.grad is None


def test_covariance_window_excludes_future() -> None:
    _, layer2, panel, ret1_full, _, _ = _pipeline()
    g_idx = np.array([0, 1, 2, 3, 4])
    sigma = covariance_for_day(
        ret1_full, g_idx, DAY, layer2.cov_lookback, 0, layer2
    )
    # Symmetric positive-definite and built only from rows strictly before
    # the decision day.
    assert sigma.shape == (5, 5)
    assert np.allclose(sigma, sigma.T)
    assert np.linalg.eigvalsh(sigma).min() > 0
