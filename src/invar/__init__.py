"""InVAR: INverted VARiate Attention for cross-sectional equity ranking.

The canonical InVAR head (bankless + regime-contrastive pretrain, locked
2026-05-19) is exposed at the top level for standalone use::

    from src.invar import InVAR, InVARConfig, train_invar

See :mod:`src.invar.canonical` for the full public API and
``docs/invar_headline_model.md`` for the model card. The historical
sub-packages ``data/``, ``model/``, ``training/``, ``evaluation/``,
``experiments/``, and ``baselines/`` predate the canonicalisation and
hold the older two-axis variant plus baseline adapters; the canonical
re-exports below are the supported entry points going forward.
"""

from src.invar.canonical import (
    InVAR,
    InVARConfig,
    PerTickerTemporalEncoder,
    MacroFiLM,
    GumbelTopKRetrievalBank,
    TemporalEncoderContrastivePretrainer,
    supcon_infonce_loss,
    pretrain_invar,
    finetune_invar,
    train_invar,
)

__all__ = [
    "InVAR",
    "InVARConfig",
    "PerTickerTemporalEncoder",
    "MacroFiLM",
    "GumbelTopKRetrievalBank",
    "TemporalEncoderContrastivePretrainer",
    "supcon_infonce_loss",
    "pretrain_invar",
    "finetune_invar",
    "train_invar",
]
