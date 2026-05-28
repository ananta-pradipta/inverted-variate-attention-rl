"""Option B: multi-task Layer-1 pretrain across universes.

Joint Stage-1 contrastive pretrain of the canonical
``PerTickerTemporalEncoder`` backbone on the UNION of multiple
universes' fold-causal training corpora, with PER-UNIVERSE input
projections that absorb the cross-universe feature-dim mismatch
(SP500/NDX = 26, NBI-enriched = 22).

The shared backbone (positional embedding, transformer encoder, layer
norm) plus the universe-specific input projection are unpacked into the
canonical ``foldF_encoder.pt`` per-(universe, fold, seed) checkpoint
consumed by the unmodified
``src.baselines.train_invar_clpretrain_v2.run_stage2_finetune`` Stage-2
loader.
"""
from src.models.multitask_l1.multitask_invar import (
    MultitaskTemporalEncoder,
    MultitaskTemporalEncoderConfig,
    UNIVERSE_FEATURE_DIMS,
    assemble_per_universe_encoder_state,
)

__all__ = [
    "MultitaskTemporalEncoder",
    "MultitaskTemporalEncoderConfig",
    "UNIVERSE_FEATURE_DIMS",
    "assemble_per_universe_encoder_state",
]
