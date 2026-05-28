"""Macro conditioning by feature-wise linear modulation (FiLM).

The macro vector is encoded, and the encoding produces a per-feature scale
gamma and shift beta applied to every per-stock token: token becomes
gamma elementwise-times token plus beta. The modulation is identity at
initialisation (gamma is 1, beta is 0) and is blended in through a learned
scalar gate initialised so the model starts as a plain ranker. The macro
encoding is also returned because Layer 3 consumes it later as a detached
regime descriptor.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from invar_rl.common.config import Layer1ModelConfig


class MacroFiLM(nn.Module):
    """Encodes the macro vector and modulates per-stock tokens."""

    def __init__(self, model_cfg: Layer1ModelConfig, macro_dim: int) -> None:
        """Initialise the FiLM block.

        Args:
            model_cfg: Layer 1 architecture configuration.
            macro_dim: Daily macro vector dimension F_macro.
        """
        super().__init__()
        d = model_cfg.d_model
        self.macro_encoder = nn.Sequential(
            nn.Linear(macro_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        # gamma and beta heads are zero-initialised so the affine map is
        # exactly identity at initialisation (gamma = 1, beta = 0).
        self.to_gamma = nn.Linear(d, d)
        self.to_beta = nn.Linear(d, d)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

        # Learned scalar blend gate; with the configured init the modulation
        # contributes nothing at the start of training.
        self.gate = nn.Parameter(
            torch.tensor(float(model_cfg.film_gate_init))
        )

    def forward(
        self, tokens: torch.Tensor, macro: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Modulate tokens and return the macro encoding.

        Args:
            tokens: Per-stock tokens, shape (n_stocks, d).
            macro: Daily macro vector, shape (F_macro,).

        Returns:
            A pair ``(modulated_tokens, macro_encoding)`` where
            ``modulated_tokens`` has shape (n_stocks, d) and
            ``macro_encoding`` has shape (d,).
        """
        macro_encoding = self.macro_encoder(macro)
        gamma = 1.0 + self.to_gamma(macro_encoding)
        beta = self.to_beta(macro_encoding)
        modulated = gamma.unsqueeze(0) * tokens + beta.unsqueeze(0)
        blended = tokens + self.gate * (modulated - tokens)
        return blended, macro_encoding
