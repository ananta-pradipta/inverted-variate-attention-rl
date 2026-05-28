"""Multi-task temporal encoder for Option B joint-universe pretrain.

Wraps the canonical ``PerTickerTemporalEncoder`` from
``src.baselines.train_invar_stx_v2`` so that the SAME shared backbone
(positional embedding, ``TransformerEncoder``, final ``LayerNorm``) is
trained across multiple universes whose per-stock feature dimensions
differ. Each universe gets its OWN ``Linear(F_universe, d_model)``
input projection; everything downstream of the projection is shared.

The shared backbone PLUS one universe's input projection can be
re-assembled into a state_dict whose keys match the canonical
``PerTickerTemporalEncoder.state_dict()`` exactly, so the produced
checkpoint loads byte-for-byte into the unmodified Stage-2 finetune
path (``run_stage2_finetune`` does a ``strict=True`` load into
``model.temporal_encoder``).

Design (Option (c) from the task spec)
--------------------------------------
The canonical encoder's only feature-dim-aware tensor is
``input_proj.weight: (d_model, F)`` (+ ``input_proj.bias: (d_model,)``).
``pos_emb``, every transformer-encoder block, and the final
``LayerNorm(d_model)`` are feature-dim agnostic. Therefore we:

  * Build N ``Linear(F_u, d_model)`` modules in a ``ModuleDict`` keyed
    by universe id; each forward pass for universe ``u`` runs
    ``self.universe_input_projs[u]`` THEN the shared backbone.
  * Hold the shared backbone as a SECOND ``PerTickerTemporalEncoder``
    instance whose ``input_proj`` is REPLACED by ``nn.Identity()``;
    its forward becomes ``Identity -> +pos_emb -> encoder -> norm[:,-1]``,
    which is what we want when the input is already in ``d_model`` space.
  * The ``assemble_per_universe_encoder_state(...)`` helper rebuilds a
    canonical-encoder state_dict for one universe by combining the
    universe-specific projection weights with the shared backbone
    weights under the canonical key names.

Leakage discipline
------------------
This module is data-agnostic: it owns NO data loaders, NO masks, NO
fold split. The fold-causal corpus restriction (pretrain_idx ==
fold_split(cfg, dates)[0]) and the leakage assertions live in the
trainer that consumes this module
(``src.baselines.train_multitask_pretrain``). The per-universe regime
fingerprint and SimCLR positive selection are computed PER UNIVERSE
inside the trainer (each universe's k-means / fingerprint stats are
fit on that universe's TRAIN-day-only data).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from src.baselines.train_invar_stx_v2 import PerTickerTemporalEncoder


# Per-universe feature dims for the three Option-B pretrain universes.
# Mirrors ``len(FEATURE_COLS)`` for each panel module:
#   sp500/lattice_native -> 26 (src.v2.data.lattice_native_panel.FEATURE_COLS)
#   nasdaq100            -> 26 (src.v2.data.nasdaq100_panel.FEATURE_COLS)
#   biotech_nbi_enriched -> 22 (src.v2.data.biotech_nbi_enriched_panel.FEATURE_COLS)
# Keys here use the same panel_kind strings the v2 runner accepts in
# ``InvarSTXV2Config.panel_kind`` so the same id flows through the
# trainer, the per-universe finetune driver, and the sbatch CLI.
UNIVERSE_FEATURE_DIMS: Dict[str, int] = {
    "lattice_native": 26,
    "nasdaq100": 26,
    "biotech_nbi_enriched": 22,
}


@dataclass
class MultitaskTemporalEncoderConfig:
    """Architecture knobs for the multi-task temporal encoder.

    Every field defaults to the canonical InVAR value so that, when
    ``panel_kind`` is held fixed, the shared backbone is shape-identical
    to ``PerTickerTemporalEncoder(...)`` used by canonical InVAR.

    ``universe_feature_dims`` keys define the universe ids that the
    encoder accepts in forward(); each key maps to that universe's
    per-stock feature count F_u (the input width of its
    ``Linear(F_u, d_model)`` projection).
    """

    temporal_window: int = 20
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    e_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    universe_feature_dims: Dict[str, int] = field(
        default_factory=lambda: dict(UNIVERSE_FEATURE_DIMS)
    )


class MultitaskTemporalEncoder(nn.Module):
    """Per-universe input projection + shared canonical backbone.

    forward(x_window, universe_id) -> (N_active, d_model).

    Shapes
    ------
    Input:
        x_window: ``(N_active, T, F_u)`` per-ticker lookback for
            universe ``universe_id`` (F_u may differ across universes).
        universe_id: one of ``self.universe_ids``; selects which
            ``Linear(F_u, d_model)`` projection to apply.
    Output:
        ``(N_active, d_model)`` last-step pooled per-ticker hiddens,
        produced by the SHARED canonical backbone after the per-universe
        projection.
    """

    def __init__(self, cfg: MultitaskTemporalEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if not cfg.universe_feature_dims:
            raise ValueError(
                "[ERR] MultitaskTemporalEncoder requires at least one "
                "universe in cfg.universe_feature_dims."
            )

        # Per-universe input projections (F_u -> d_model). Use a
        # ModuleDict keyed by the panel_kind string so the trainer can
        # look projections up by universe id at every forward call.
        self.universe_input_projs = nn.ModuleDict()
        for uid, fdim in cfg.universe_feature_dims.items():
            if not isinstance(fdim, int) or fdim < 1:
                raise ValueError(
                    f"[ERR] universe_feature_dims[{uid}] must be a "
                    f"positive int; got {fdim!r}."
                )
            self.universe_input_projs[uid] = nn.Linear(fdim, cfg.d_model)

        # SHARED backbone: a canonical PerTickerTemporalEncoder whose
        # input_proj is REPLACED by Identity. Its remaining tensors
        # (pos_emb, encoder.*, norm.*) are the shared backbone state
        # the multi-universe pretrain optimises jointly.
        self.shared_backbone = PerTickerTemporalEncoder(
            n_features=cfg.d_model,  # placeholder; Identity overrides
            temporal_window=cfg.temporal_window,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            e_layers=cfg.e_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )
        # Strip the canonical Linear(F, d_model) input_proj; downstream
        # state-dict assembly relies on this being an Identity so the
        # shared backbone holds NO universe-specific weights.
        self.shared_backbone.input_proj = nn.Identity()

    @property
    def universe_ids(self) -> List[str]:
        """Ordered list of universe ids this encoder accepts."""
        return list(self.universe_input_projs.keys())

    def forward(self, x_window: Tensor, universe_id: str) -> Tensor:
        """Encode ``(N_active, T, F_u)`` -> ``(N_active, d_model)``.

        Args:
            x_window: per-ticker lookback windows for universe
                ``universe_id``. The last-dim width ``F_u`` MUST match
                the registered ``universe_feature_dims[universe_id]``.
            universe_id: which universe's projection to apply.

        Returns:
            ``(N_active, d_model)`` last-step pooled per-ticker hiddens
            from the shared backbone.
        """
        if universe_id not in self.universe_input_projs:
            raise KeyError(
                f"[ERR] unknown universe_id={universe_id!r}; "
                f"registered: {self.universe_ids}"
            )
        expected_f = int(self.cfg.universe_feature_dims[universe_id])
        if x_window.dim() != 3 or int(x_window.shape[-1]) != expected_f:
            raise ValueError(
                f"[ERR] universe={universe_id} expects "
                f"x_window shape (N, T, {expected_f}); got "
                f"tuple({tuple(x_window.shape)})."
            )
        proj = self.universe_input_projs[universe_id]
        # (N, T, F_u) -> (N, T, d_model) projected tokens.
        h_proj = proj(x_window)
        # Shared backbone consumes (N, T, d_model) and returns (N, d).
        # The Identity input_proj passes h_proj through untouched; the
        # canonical pos_emb + transformer + norm + last-step pool then
        # apply unchanged.
        return self.shared_backbone(h_proj)

    def day_embedding(
        self,
        x_window: Tensor,
        universe_id: str,
        proj_head: nn.Module,
    ) -> Tensor:
        """Mask-mean-pool active tickers and project to the SimCLR space.

        Mirrors
        ``TemporalEncoderContrastivePretrainer.day_embedding`` from
        the canonical clpretrain trainer (mean-pool over active tickers
        then projection head then L2 normalise) but parameterises the
        encoder over the universe id so the same call site can produce
        day embeddings for every universe in the multi-task batch.

        Args:
            x_window: ``(N_active, T, F_u)`` active-ticker windows for
                a single training day in universe ``universe_id``.
            universe_id: which universe's input projection to use.
            proj_head: external SimCLR projection head (NOT owned by
                this module so callers can share / serialise it on the
                same schedule as their pretrain loop).

        Returns:
            ``(proj_dim,)`` L2-normalised projected day embedding.
        """
        per_ticker = self.forward(x_window, universe_id)  # (N_active, d)
        day_vec = per_ticker.mean(dim=0)                  # (d,)
        z = proj_head(day_vec)                            # (proj_dim,)
        return torch.nn.functional.normalize(z, dim=-1)


def assemble_per_universe_encoder_state(
    multitask: MultitaskTemporalEncoder,
    universe_id: str,
) -> Dict[str, Tensor]:
    """Build a canonical ``PerTickerTemporalEncoder.state_dict()`` for
    one universe by combining the shared backbone with that universe's
    input projection.

    The returned mapping uses the SAME key names a freshly constructed
    canonical encoder would emit (``input_proj.weight``,
    ``input_proj.bias``, ``pos_emb``, ``encoder.*``, ``norm.*``), so the
    Stage-2 finetune loader's strict load into ``model.temporal_encoder``
    works byte-for-byte unchanged.

    Args:
        multitask: trained ``MultitaskTemporalEncoder``.
        universe_id: which universe's input projection to attach. MUST
            be a registered key in ``multitask.universe_input_projs``.

    Returns:
        Detached CPU tensors keyed by canonical state-dict names. Safe
        to wrap in the canonical ``foldF_encoder.pt`` payload directly.
    """
    if universe_id not in multitask.universe_input_projs:
        raise KeyError(
            f"[ERR] unknown universe_id={universe_id!r}; "
            f"registered: {multitask.universe_ids}"
        )
    proj = multitask.universe_input_projs[universe_id]
    # Shared backbone state dict; strip the Identity input_proj entries
    # (there are none, since nn.Identity has no parameters, but we keep
    # this explicit so the assembly logic is auditable).
    backbone_state = multitask.shared_backbone.state_dict()
    out: Dict[str, Tensor] = {}
    for k, v in backbone_state.items():
        if k.startswith("input_proj"):
            # Defensive: nn.Identity has no params; if a future refactor
            # introduces some, drop them here so the canonical key space
            # is owned exclusively by the universe-specific projection
            # below.
            continue
        out[k] = v.detach().cpu().clone()
    # Universe-specific projection (canonical key names).
    out["input_proj.weight"] = proj.weight.detach().cpu().clone()
    out["input_proj.bias"] = proj.bias.detach().cpu().clone()
    return out


def expected_canonical_encoder_keys(
    n_features: int,
    cfg: MultitaskTemporalEncoderConfig,
) -> List[str]:
    """Return the canonical-encoder state_dict keys for one universe.

    Used by unit tests and the assemble helper's invariant checks. The
    keys come from a freshly constructed ``PerTickerTemporalEncoder``
    with the requested universe feature dim; the canonical Stage-2
    loader expects exactly this set.
    """
    canonical = PerTickerTemporalEncoder(
        n_features=n_features,
        temporal_window=cfg.temporal_window,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        e_layers=cfg.e_layers,
        dropout=cfg.dropout,
        activation=cfg.activation,
    )
    return sorted(canonical.state_dict().keys())
