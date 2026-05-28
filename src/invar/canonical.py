"""Canonical InVAR public API.

This module is the single import surface for the canonical InVAR
("INverted VARiate Attention") model, the bankless + regime-contrastive
pretrain ("clpretrain") variant locked as the project default on
2026-05-19. See ``docs/invar_headline_model.md`` and the memory entry
``invar-canonical-2026-05-19`` for the full identity.

Use this module when you need to instantiate or train InVAR as a
standalone ranker, decoupled from the InVAR-RL three-layer stack.

Quick reference::

    import torch
    from pathlib import Path
    from src.invar.canonical import InVARConfig, train_invar

    cfg = InVARConfig(fold=1, seed=42)
    cfg.panel_kind = "lattice_native"
    cfg.two_regime_val = True
    cfg.output_dir = "results/invar_run"
    cfg.panel_end = "2025-12-31"

    train_invar(
        cfg=cfg,
        ckpt_path=Path(cfg.output_dir) / "_ckpt" / f"fold{cfg.fold}_encoder.pt",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

The classes and functions below are re-exports from the canonical
implementation in ``src/baselines/train_invar_stx_v2.py`` and
``src/baselines/train_invar_clpretrain_v2.py``. The legacy paths
continue to work; this module exists so consumers (InVAR-RL,
notebooks, ad-hoc experiments) can depend on a stable name without
reaching into ``src/baselines/``.

The headline result this configuration reproduces (5 folds, 5 seeds,
universal S&P 500 ``lattice_native`` panel, two-regime val):

============  =============  =====
fold          rank IC mean   std
============  =============  =====
F1 covid      +0.0552        0.011
F2 rate-str   +0.0028        0.007
F3 post-str   +0.0349        0.005
F4 ai-rally   +0.0305        0.003
F5 fed-cut    +0.0186        0.003
pooled (25)   +0.0284        0.019
============  =============  =====
"""

from __future__ import annotations

from src.baselines.train_invar_stx_v2 import (
    GumbelTopKRetrievalBank,
    InvarSTXModel as InVAR,
    InvarSTXV2Config as InVARConfig,
    MacroFiLM,
    PerTickerTemporalEncoder,
)
from src.baselines.train_invar_clpretrain_v2 import (
    TemporalEncoderContrastivePretrainer,
    _supcon_infonce_loss as supcon_infonce_loss,
    run_stage1_pretrain as pretrain_invar,
    run_stage1_sequential_pretrain as pretrain_invar_sequential,
    run_stage2_finetune as finetune_invar,
)


def train_invar(
    cfg: InVARConfig,
    ckpt_path,
    device,
    pretrain_epochs: int = 10,
    finetune_epochs: int = 10,
    pretrain_only: bool = False,
    skip_pretrain: bool = False,
) -> None:
    """Run the canonical two-stage InVAR training in one call.

    Convenience wrapper over :func:`pretrain_invar` (fold-causal
    regime-contrastive pretrain on the fold's training corpus) and
    :func:`finetune_invar` (per-seed supervised finetune from the
    fold's pretrain checkpoint, layer-wise LR 0.25x on the loaded
    encoder).

    The fold and seed are read from ``cfg.fold`` and ``cfg.seed``;
    setting them externally lets callers run the same training with
    different finetune seeds against a single shared per-fold
    pretrain checkpoint, matching the production sbatch flow in
    ``scripts/wulver/invar_clpretrain.sbatch``.

    Args:
        cfg: :class:`InVARConfig` with at least ``fold``, ``seed``,
            ``panel_kind``, ``two_regime_val``, ``output_dir``,
            ``panel_end`` set.
        ckpt_path: Path object for the per-fold encoder checkpoint
            (typically ``Path(cfg.output_dir) / "_ckpt" /
            f"fold{cfg.fold}_encoder.pt"``).
        device: ``torch.device`` for training (cuda or cpu).
        pretrain_epochs: Epochs for the regime-contrastive pretrain
            stage; ignored when ``skip_pretrain`` is True.
        finetune_epochs: Epochs for the supervised finetune stage;
            ignored when ``pretrain_only`` is True.
        pretrain_only: If True, run only the pretrain stage and exit.
        skip_pretrain: If True, expect ``ckpt_path`` to already exist
            and skip directly to finetune.
    """
    if skip_pretrain and pretrain_only:
        raise ValueError(
            "pretrain_only and skip_pretrain are mutually exclusive."
        )
    assert cfg.enable_retrieval_bank is False, (
        "Canonical InVAR is BANKLESS; "
        "cfg.enable_retrieval_bank must be False."
    )
    if not skip_pretrain:
        # A1 (2026-05-27): sequential wrapper preserves the canonical
        # single-stage path byte-identically when
        # cfg.pretrain_stages == ["regime"] (the default).
        pretrain_invar_sequential(
            cfg, pretrain_epochs, device, ckpt_path,
        )
    if pretrain_only:
        return
    finetune_invar(cfg, finetune_epochs, device, ckpt_path)


__all__ = [
    "InVAR",
    "InVARConfig",
    "PerTickerTemporalEncoder",
    "MacroFiLM",
    "GumbelTopKRetrievalBank",
    "TemporalEncoderContrastivePretrainer",
    "supcon_infonce_loss",
    "pretrain_invar",
    "pretrain_invar_sequential",
    "finetune_invar",
    "train_invar",
]
