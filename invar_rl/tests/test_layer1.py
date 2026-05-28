"""Tests for Layer 1, the InVAR ranker."""

from __future__ import annotations

from pathlib import Path

import torch

from invar_rl.common.config import load_base_config, load_layer1_config
from invar_rl.common.seeding import set_global_seed
from invar_rl.data.synthetic import SyntheticPanel
from invar_rl.layer1_ranker.invar import INVAR

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _model_and_panel() -> tuple[INVAR, SyntheticPanel]:
    base = load_base_config(CONFIG_DIR / "base.yaml")
    layer1 = load_layer1_config(CONFIG_DIR / "layer1.yaml")
    set_global_seed(42)
    model = INVAR(
        layer1.model,
        n_features=base.synthetic.n_features,
        lookback=base.synthetic.lookback,
        macro_dim=base.synthetic.macro_dim,
    )
    panel = SyntheticPanel(base.synthetic, seed=42)
    return model, panel


def _day_tensors(panel: SyntheticPanel, day: int):
    _, feats = panel.feature_window(day)
    macro = panel.macro_vector(day)
    return (
        torch.from_numpy(feats).float(),
        torch.from_numpy(macro).float(),
    )


def test_forward_shapes() -> None:
    model, panel = _model_and_panel()
    feats, macro = _day_tensors(panel, 400)
    out = model(feats, macro)
    n_active = feats.shape[0]
    assert out.scores.shape == (n_active,)
    assert out.macro_regime_encoding.shape == (
        load_layer1_config(CONFIG_DIR / "layer1.yaml").model.d_model,
    )
    assert "score_dispersion" in out.summary
    assert out.summary["score_dispersion"].ndim == 0


def test_handles_variable_active_count() -> None:
    model, panel = _model_and_panel()
    out_early = model(*_day_tensors(panel, 60))
    out_late = model(*_day_tensors(panel, 1200))
    n_early = len(panel.active_tickers(60))
    n_late = len(panel.active_tickers(1200))
    assert out_early.scores.shape == (n_early,)
    assert out_late.scores.shape == (n_late,)
    assert n_late != n_early


def test_film_is_identity_at_initialisation() -> None:
    model, panel = _model_and_panel()
    model.eval()
    feats, macro = _day_tensors(panel, 400)
    tokens = model.encoder(feats)
    modulated, _ = model.film(tokens, macro)
    assert torch.allclose(modulated, tokens, atol=1e-6)


def test_gradients_reach_all_parameters() -> None:
    model, panel = _model_and_panel()
    model.train()
    feats, macro = _day_tensors(panel, 400)
    out = model(feats, macro)
    out.scores.sum().backward()
    missing = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing, f"parameters without gradient: {missing}"
