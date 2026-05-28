"""Canonical InVAR adapter for InVAR-RL Layer 1.

This module wraps the canonical InVAR ranker (``src.invar.InVAR``,
the bankless + regime-contrastive pretrain variant locked
2026-05-19, see ``docs/invar_headline_model.md``) so it can be
dropped into InVAR-RL's Layer 1 slot without duplicating model
code. The original Layer 1 module in this branch
(:mod:`invar_rl.layer1_ranker.invar`) is a stripped FiLM +
cross-stock-attention skeleton; this adapter routes Layer 1 to
the real canonical InVAR instead.

The adapter exposes the same :class:`Layer1Output` contract that
Layer 2 (QP) and Layer 3 (RL) already consume (``scores``,
``macro_regime_encoding``, ``summary``). Internally it builds the
eight inputs that ``InvarSTXModel.forward`` requires (x_window,
day_query_key, query_day_idx, allowed_day_indices, regime_scalars,
duration_input, macro_input, macro_gate_input) from a single
batch dict produced by the panel loader.

Status (2026-05-19): adapter scaffold. Stage 1 of InVAR-RL is
expected to delegate the actual training to
``src.baselines.train_invar_clpretrain_v2`` (one job per fold via
``scripts/wulver/invar_clpretrain.sbatch``), which produces both
the per-fold encoder checkpoint and a fully-trained per-seed
finetune checkpoint that this adapter then loads. The data-pipeline
bridge that produces the 8-input batch dict is wired by the
follow-up commit; this file defines the wrapper class so Layer 2
and Layer 3 already see the canonical InVAR's outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

from src.invar import InVAR, InVARConfig

from invar_rl.layer1_ranker.invar import Layer1Output


@dataclass
class CanonicalInVARConfig:
    """Adapter-side knobs that select which canonical InVAR ckpt to load
    and how to expose the regime descriptor to Layer 3.

    Most architectural knobs live on :class:`src.invar.InVARConfig`; the
    fields here only control how this adapter consumes the canonical
    ranker, not the ranker itself.
    """

    panel_kind: str = "lattice_native"
    two_regime_val: bool = True
    panel_end: str = "2025-12-31"
    output_dir: str = "results/invar_clpretrain"
    regime_descriptor_dim: int = 32
    macro_regime_source: str = "macro_input"


class CanonicalInVARLayer1(nn.Module):
    """InVAR-RL Layer 1 backed by canonical InVAR.

    The constructor takes the canonical :class:`InVARConfig` and, when
    a checkpoint path is given, loads the per-seed finetune weights
    produced by the canonical training pipeline. The forward signature
    matches what Layer 2 and Layer 3 expect: a single ``batch`` dict
    containing the canonical InVAR's eight inputs for one trading day.

    Args:
        cfg: :class:`InVARConfig` instance describing the canonical
            backbone (fold, seed, panel knobs, architectural knobs).
        adapter_cfg: :class:`CanonicalInVARConfig` with adapter-side
            options (regime descriptor source, output paths).
        ckpt_path: Optional path to a per-seed finetune checkpoint.
            When provided, weights are loaded with strict key match
            and the wrapper is set to eval mode; Layer 2 / Layer 3
            then consume it as a frozen scorer.
    """

    def __init__(
        self,
        cfg: InVARConfig,
        adapter_cfg: Optional[CanonicalInVARConfig] = None,
        ckpt_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        assert cfg.enable_retrieval_bank is False, (
            "Canonical InVAR is bankless; "
            "cfg.enable_retrieval_bank must be False."
        )
        self.cfg = cfg
        self.adapter_cfg = adapter_cfg or CanonicalInVARConfig()
        self.model = InVAR(cfg)
        if ckpt_path is not None:
            blob = torch.load(ckpt_path, map_location="cpu")
            state = blob.get("state_dict", blob)
            self.model.load_state_dict(state, strict=True)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, batch: Dict[str, torch.Tensor]) -> Layer1Output:
        """Score one trading day through the canonical InVAR.

        Args:
            batch: A dict with the eight canonical InVAR inputs for one
                day, produced by the panel-loader bridge. Keys:
                ``x_window`` (N, T, F), ``day_query_key`` (14,),
                ``query_day_idx`` (int), ``allowed_day_indices`` (1d),
                ``regime_scalars`` (2,), ``duration_input`` (N, D_dur),
                ``macro_input`` (D_macro,), ``macro_gate_input`` (D_gate,).

        Returns:
            :class:`Layer1Output` with the canonical scores, a regime
            descriptor sourced from the macro input (default;
            configurable via ``adapter_cfg.macro_regime_source``), and
            the standard summary stats.
        """
        scores = self.model(
            x_window=batch["x_window"],
            day_query_key=batch["day_query_key"],
            query_day_idx=int(batch["query_day_idx"]),
            allowed_day_indices=batch["allowed_day_indices"],
            regime_scalars=batch["regime_scalars"],
            duration_input=batch["duration_input"],
            macro_input=batch["macro_input"],
            macro_gate_input=batch["macro_gate_input"],
        )
        if self.adapter_cfg.macro_regime_source == "macro_input":
            macro_regime_encoding = batch["macro_input"].detach()
        elif self.adapter_cfg.macro_regime_source == "macro_gate_input":
            macro_regime_encoding = batch["macro_gate_input"].detach()
        else:
            raise ValueError(
                f"unknown macro_regime_source: "
                f"{self.adapter_cfg.macro_regime_source}"
            )
        summary = {
            "score_dispersion": scores.std(unbiased=False),
            "n_active": torch.tensor(
                scores.numel(), dtype=torch.long, device=scores.device
            ),
        }
        return Layer1Output(
            scores=scores,
            macro_regime_encoding=macro_regime_encoding,
            summary=summary,
        )


__all__ = [
    "CanonicalInVARConfig",
    "CanonicalInVARLayer1",
]
