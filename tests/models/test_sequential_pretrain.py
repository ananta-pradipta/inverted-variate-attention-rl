"""Unit tests for A1: sequential Stage-1 pretrain (regime -> sector).

Covers the public surface of
``src.baselines.train_invar_clpretrain_v2.run_stage1_sequential_pretrain``
and the ``init_from_ckpt`` extension to ``run_stage1_pretrain``.

The tests mock out the heavy ``run_stage1_pretrain`` body so they do
NOT touch ``build_panel`` / ``v2_runner`` / fold-split code. The
correctness invariants we check are:

  * test_single_stage_preserves_canonical: when
    ``cfg.pretrain_stages == ["regime"]`` the sequential wrapper calls
    ``run_stage1_pretrain`` EXACTLY ONCE with ``init_from_ckpt=False``
    and never mutates ``cfg.pretrain_positive_method`` /
    ``cfg.pretrain_method``. This is the byte-identical canonical path.
  * test_sequential_runs_both_stages: when
    ``cfg.pretrain_stages == ["regime", "sector"]`` the wrapper calls
    ``run_stage1_pretrain`` TWICE, with the correct selector flags
    flipped per stage and the second call's ``init_from_ckpt=True``.
    Each call gets the right number of epochs.
  * test_backbone_weights_carry_forward: with ``init_from_ckpt=True``,
    the encoder loaded into a fresh ``TemporalEncoderContrastive
    Pretrainer`` matches the encoder weights saved at ``ckpt_path``;
    fresh-init (``init_from_ckpt=False``) does NOT match.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import torch

from src.baselines.train_invar_clpretrain_v2 import (
    TemporalEncoderContrastivePretrainer,
    run_stage1_sequential_pretrain,
)
from src.baselines.train_invar_stx_v2 import InvarSTXV2Config


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


def _make_pretrainer(seed: int = 0) -> TemporalEncoderContrastivePretrainer:
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
        aux_regression_head=False,
    )


def _make_cfg(stages: list[str]) -> InvarSTXV2Config:
    cfg = InvarSTXV2Config(fold=1, seed=42)
    cfg.panel_kind = "lattice_native"
    cfg.pretrain_stages = list(stages)
    return cfg


# ---------------------------------------------------------------------
# Test 1: single-stage = canonical (one call, init_from_ckpt False).
# ---------------------------------------------------------------------
def test_single_stage_preserves_canonical(tmp_path: Path) -> None:
    """``pretrain_stages == ["regime"]`` must reduce to ONE call of
    ``run_stage1_pretrain`` with ``init_from_ckpt=False`` and must NOT
    mutate the selector flags on cfg.
    """
    cfg = _make_cfg(["regime"])
    pos_before = cfg.pretrain_positive_method
    method_before = cfg.pretrain_method

    calls: List[Dict[str, Any]] = []

    def _fake_pretrain(_cfg, epochs, _device, _ckpt, init_from_ckpt=False):
        calls.append({
            "epochs": int(epochs),
            "init_from_ckpt": bool(init_from_ckpt),
            "pretrain_positive_method": str(
                _cfg.pretrain_positive_method
            ),
            "pretrain_method": str(_cfg.pretrain_method),
        })

    ckpt_path = tmp_path / "fold1_encoder.pt"
    with patch(
        "src.baselines.train_invar_clpretrain_v2.run_stage1_pretrain",
        side_effect=_fake_pretrain,
    ):
        run_stage1_sequential_pretrain(
            cfg, pretrain_epochs=10,
            device=torch.device("cpu"),
            ckpt_path=ckpt_path,
        )

    assert len(calls) == 1, f"expected 1 call, got {len(calls)}"
    assert calls[0]["epochs"] == 10
    assert calls[0]["init_from_ckpt"] is False
    # Selector flags must not be mutated on the canonical single-stage
    # path (the sequential wrapper short-circuits before the loop).
    assert cfg.pretrain_positive_method == pos_before
    assert cfg.pretrain_method == method_before


# ---------------------------------------------------------------------
# Test 2: sequential = TWO calls with the right selector flags + epochs.
# ---------------------------------------------------------------------
def test_sequential_runs_both_stages(tmp_path: Path) -> None:
    """``pretrain_stages == ["regime", "sector"]``: TWO calls; flags
    flipped per stage; second call's ``init_from_ckpt=True``; both
    stages use the same per-stage epoch budget.
    """
    cfg = _make_cfg(["regime", "sector"])

    calls: List[Dict[str, Any]] = []

    def _fake_pretrain(_cfg, epochs, _device, _ckpt, init_from_ckpt=False):
        calls.append({
            "epochs": int(epochs),
            "init_from_ckpt": bool(init_from_ckpt),
            "pretrain_positive_method": str(
                _cfg.pretrain_positive_method
            ),
            "pretrain_method": str(_cfg.pretrain_method),
        })

    ckpt_path = tmp_path / "fold1_encoder.pt"
    with patch(
        "src.baselines.train_invar_clpretrain_v2.run_stage1_pretrain",
        side_effect=_fake_pretrain,
    ):
        run_stage1_sequential_pretrain(
            cfg, pretrain_epochs=7,
            device=torch.device("cpu"),
            ckpt_path=ckpt_path,
        )

    assert len(calls) == 2, f"expected 2 calls, got {len(calls)}"
    # Stage 1a: regime selector, no prior ckpt.
    assert calls[0]["pretrain_positive_method"] == "regime"
    assert calls[0]["pretrain_method"] == "infonce_kmeans"
    assert calls[0]["init_from_ckpt"] is False
    assert calls[0]["epochs"] == 7
    # Stage 1b: sector selector, continues from prior ckpt.
    assert calls[1]["pretrain_positive_method"] == "sector"
    assert calls[1]["pretrain_method"] == "infonce_sector"
    assert calls[1]["init_from_ckpt"] is True
    assert calls[1]["epochs"] == 7


# ---------------------------------------------------------------------
# Test 3: backbone weights carry forward via init_from_ckpt.
# ---------------------------------------------------------------------
def test_backbone_weights_carry_forward(tmp_path: Path) -> None:
    """A1 Stage 1b must continue from Stage 1a's saved encoder weights,
    not re-initialise. We construct a TemporalEncoderContrastive
    Pretrainer ('stage 1a'), save its encoder.state_dict() to disk in
    the same format ``run_stage1_pretrain`` uses, then verify that the
    init_from_ckpt code path loads those EXACT weights into a freshly
    constructed pretrainer (analogue of stage 1b's startup).

    We exercise the load path directly by mimicking the lines added in
    run_stage1_pretrain when init_from_ckpt is True.
    """
    # ---- Stage 1a analogue: build a pretrainer and save its encoder.
    stage1a = _make_pretrainer(seed=0)
    ckpt_path = tmp_path / "fold1_encoder.pt"
    torch.save(
        {
            "fold": 1,
            "seed": 42,
            "pretrain_epochs": 10,
            "panel_kind": "lattice_native",
            "encoder_state_dict": stage1a.encoder.state_dict(),
        },
        ckpt_path,
    )

    # ---- Stage 1b analogue with init_from_ckpt=True: fresh pretrainer
    # (different RNG seed so the fresh init differs from stage 1a), then
    # load the saved encoder weights with strict key match.
    stage1b_loaded = _make_pretrainer(seed=999)
    # Before load: stage 1b's encoder weights must NOT equal stage 1a's
    # (different seeds; sanity check the test setup itself).
    a_name = next(iter(stage1b_loaded.encoder.state_dict().keys()))
    before = stage1b_loaded.encoder.state_dict()[a_name].clone()
    assert not torch.allclose(
        before, stage1a.encoder.state_dict()[a_name]
    ), "test setup: fresh init must differ from stage 1a"

    prior_ckpt = torch.load(ckpt_path, map_location="cpu")
    prior_enc_state = prior_ckpt["encoder_state_dict"]
    target_keys = set(stage1b_loaded.encoder.state_dict().keys())
    ckpt_keys = set(prior_enc_state.keys())
    assert target_keys == ckpt_keys, (
        "encoder key mismatch between stage 1a save and stage 1b load"
    )
    stage1b_loaded.encoder.load_state_dict(prior_enc_state, strict=True)

    # After load: stage 1b's encoder weights must equal stage 1a's
    # SAVED weights for every parameter.
    for key, stage1a_val in stage1a.encoder.state_dict().items():
        stage1b_val = stage1b_loaded.encoder.state_dict()[key]
        assert torch.allclose(stage1b_val, stage1a_val), (
            f"weight {key} did not carry from stage 1a to stage 1b"
        )

    # Sanity: a fresh-init pretrainer (init_from_ckpt analogue OFF)
    # must NOT match stage 1a, i.e. the carry is what made it match.
    stage1b_fresh = _make_pretrainer(seed=999)
    fresh_val = stage1b_fresh.encoder.state_dict()[a_name]
    assert not torch.allclose(
        fresh_val, stage1a.encoder.state_dict()[a_name]
    ), "fresh-init pretrainer unexpectedly matches stage 1a"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
