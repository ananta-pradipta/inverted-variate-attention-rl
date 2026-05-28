"""Shared portfolio summary-statistics interface.

Both the QP layer and the soft top-k layer return this exact dictionary so
the two allocation constructions are interchangeable downstream (Layer 3
consumes these as detached observations later).
"""

from __future__ import annotations

from typing import Dict

import torch


def portfolio_summary(
    weights: torch.Tensor, sigma: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Compute the shared summary statistics for a weight vector.

    Args:
        weights: Portfolio weights, shape (n,).
        sigma: Covariance used for the predicted-volatility term, shape
            (n, n).

    Returns:
        A dictionary with scalar tensors ``predicted_vol`` (square root of
        w' Sigma w), ``effective_positions`` (participation ratio
        (sum |w|)^2 / sum w^2), ``gross_exposure`` (sum |w|), and
        ``net_exposure`` (sum w).
    """
    gross = weights.abs().sum()
    variance = weights @ (sigma @ weights)
    predicted_vol = torch.sqrt(torch.clamp(variance, min=0.0))
    sq = (weights ** 2).sum()
    effective = torch.where(
        sq > 0, gross ** 2 / sq, weights.new_zeros(())
    )
    return {
        "predicted_vol": predicted_vol,
        "effective_positions": effective,
        "gross_exposure": gross,
        "net_exposure": weights.sum(),
    }
