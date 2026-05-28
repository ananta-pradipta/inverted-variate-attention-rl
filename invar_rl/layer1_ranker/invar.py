"""Full Layer 1 model: the InVAR ranker.

Composition order per day: per-stock temporal encoder, FiLM macro
conditioning, inverted cross-stock attention, then a linear score head. The
forward pass returns the score vector, the macro-regime encoding, and a small
dictionary of summary statistics. The return signature is designed so Layer 3
can consume the macro encoding and the summary statistics later as detached
observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from invar_rl.common.config import Layer1ModelConfig
from invar_rl.layer1_ranker.cross_attention import InvertedCrossStockAttention
from invar_rl.layer1_ranker.encoder import PerStockTemporalEncoder
from invar_rl.layer1_ranker.film import MacroFiLM


@dataclass
class Layer1Output:
    """Per-day output of the InVAR ranker.

    Attributes:
        scores: Scalar score per active stock, shape (n_stocks,).
        macro_regime_encoding: Encoding of the macro vector, shape (d,).
            Consumed by Layer 3 as a detached regime descriptor.
        summary: Scalar summary statistics. Contains at least
            ``score_dispersion`` (the cross-sectional standard deviation of
            the scores) and ``n_active``.
    """

    scores: torch.Tensor
    macro_regime_encoding: torch.Tensor
    summary: Dict[str, torch.Tensor]


class INVAR(nn.Module):
    """Inverted-variate-attention cross-sectional ranker."""

    def __init__(
        self,
        model_cfg: Layer1ModelConfig,
        n_features: int,
        lookback: int,
        macro_dim: int,
    ) -> None:
        """Initialise the full ranker.

        Args:
            model_cfg: Layer 1 architecture configuration.
            n_features: Per-stock feature count F.
            lookback: Window length L.
            macro_dim: Daily macro vector dimension F_macro.
        """
        super().__init__()
        self.encoder = PerStockTemporalEncoder(
            model_cfg, n_features=n_features, lookback=lookback
        )
        self.film = MacroFiLM(model_cfg, macro_dim=macro_dim)
        self.cross_attention = InvertedCrossStockAttention(model_cfg)
        self.score_head = nn.Linear(model_cfg.d_model, 1)

    def forward(
        self, features: torch.Tensor, macro: torch.Tensor
    ) -> Layer1Output:
        """Score every active stock for one day.

        Args:
            features: Lookback windows, shape (n_stocks, L, F).
            macro: Daily macro vector, shape (F_macro,).

        Returns:
            A ``Layer1Output``.
        """
        tokens = self.encoder(features)
        tokens, macro_encoding = self.film(tokens, macro)
        tokens = self.cross_attention(tokens)
        scores = self.score_head(tokens).squeeze(-1)

        n_active = scores.shape[0]
        dispersion = (
            scores.std(unbiased=False)
            if n_active > 1
            else scores.new_zeros(())
        )
        summary = {
            "score_dispersion": dispersion,
            "n_active": scores.new_tensor(float(n_active)),
        }
        return Layer1Output(
            scores=scores,
            macro_regime_encoding=macro_encoding,
            summary=summary,
        )
