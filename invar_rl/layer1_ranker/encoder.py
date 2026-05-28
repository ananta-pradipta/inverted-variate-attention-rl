"""Per-stock temporal encoder.

For each active stock, take the (L, F) lookback window, project to the model
dimension, add a learned positional embedding over the L steps, pass through a
transformer encoder stack, and pool the last time step to one token per stock.
"""

from __future__ import annotations

import torch
from torch import nn

from invar_rl.common.config import Layer1ModelConfig


class PerStockTemporalEncoder(nn.Module):
    """Encodes one (n_stocks, L, F) tensor into (n_stocks, d) tokens."""

    def __init__(
        self, model_cfg: Layer1ModelConfig, n_features: int, lookback: int
    ) -> None:
        """Initialise the encoder.

        Args:
            model_cfg: Layer 1 architecture configuration.
            n_features: Per-stock feature count F.
            lookback: Window length L.
        """
        super().__init__()
        d = model_cfg.d_model
        self._lookback = lookback
        self.input_proj = nn.Linear(n_features, d)
        self.pos_embedding = nn.Parameter(torch.zeros(1, lookback, d))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=model_cfg.encoder_heads,
            dim_feedforward=model_cfg.feedforward,
            dropout=model_cfg.dropout,
            activation=model_cfg.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=model_cfg.encoder_layers
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Encode lookback windows.

        Args:
            windows: Tensor of shape (n_stocks, L, F).

        Returns:
            Tensor of shape (n_stocks, d), the last-step pooled token per
            stock.
        """
        if windows.dim() != 3:
            raise ValueError(
                f"expected (n_stocks, L, F), got shape {tuple(windows.shape)}"
            )
        if windows.shape[1] != self._lookback:
            raise ValueError(
                f"window length {windows.shape[1]} does not match configured "
                f"lookback {self._lookback}"
            )
        h = self.input_proj(windows) + self.pos_embedding
        h = self.encoder(h)
        return h[:, -1, :]
