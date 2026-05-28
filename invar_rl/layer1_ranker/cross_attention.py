"""Inverted cross-stock attention.

Take the set of per-stock tokens for one day, treat the stock axis as the
sequence axis, and run a small transformer encoder so dense self-attention
operates across stocks within the day. There is no graph and no adjacency.
"""

from __future__ import annotations

import torch
from torch import nn

from invar_rl.common.config import Layer1ModelConfig


class InvertedCrossStockAttention(nn.Module):
    """Dense self-attention over the stock axis within a single day."""

    def __init__(self, model_cfg: Layer1ModelConfig) -> None:
        """Initialise the inverted attention block.

        Args:
            model_cfg: Layer 1 architecture configuration.
        """
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=model_cfg.d_model,
            nhead=model_cfg.cross_attention_heads,
            dim_feedforward=model_cfg.feedforward,
            dropout=model_cfg.dropout,
            activation=model_cfg.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=model_cfg.cross_attention_layers
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mix information across stocks.

        Args:
            tokens: Tensor of shape (n_stocks, d).

        Returns:
            Tensor of shape (n_stocks, d) after cross-stock attention.
        """
        if tokens.dim() != 2:
            raise ValueError(
                f"expected (n_stocks, d), got shape {tuple(tokens.shape)}"
            )
        # The stock axis is the sequence; one day is one batch element.
        mixed = self.encoder(tokens.unsqueeze(0))
        return mixed.squeeze(0)
