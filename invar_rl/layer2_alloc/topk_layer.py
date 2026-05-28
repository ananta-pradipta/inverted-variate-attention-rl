"""Differentiable soft top-k long-short layer (the ablation allocation).

Forms a long-short basket directly from the score vector using a
temperature-controlled smooth relaxation of selecting the highest-scoring k
and the lowest-scoring k stocks, then normalises to a dollar-neutral,
unit-gross book. The temperature is configurable and annealable. The layer
returns weights and the same summary-statistics dictionary as the QP layer,
so the two are interchangeable downstream.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from invar_rl.common.config import Layer2Config
from invar_rl.layer2_alloc.summary import portfolio_summary

_EPS = 1e-8


class SoftTopKLongShort:
    """Smooth top-k long-short allocator."""

    def __init__(self, cfg: Layer2Config) -> None:
        """Initialise the allocator.

        Args:
            cfg: Layer 2 configuration providing k, the initial temperature,
                the anneal flag, and the gross-leverage cap.
        """
        self._k = int(cfg.topk_k)
        self._temperature = float(cfg.topk_temperature)
        self._anneal = bool(cfg.topk_temperature_anneal)
        self._gross = float(cfg.gross_leverage)

    @property
    def temperature(self) -> float:
        return self._temperature

    def set_temperature(self, value: float) -> None:
        """Set the relaxation temperature (used for annealing schedules)."""
        if value <= 0.0:
            raise ValueError("temperature must be positive")
        self._temperature = float(value)

    @property
    def anneal(self) -> bool:
        return self._anneal

    def __call__(
        self, scores: torch.Tensor, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Form the long-short book for one day.

        Args:
            scores: Score vector s, shape (n,).
            sigma: Covariance for the predicted-volatility summary, shape
                (n, n).

        Returns:
            A pair ``(weights, summary)`` with the shared summary interface.
        """
        if scores.dim() != 1:
            raise ValueError(
                f"scores must be 1-D, got shape {tuple(scores.shape)}"
            )
        n = scores.shape[0]
        k = min(self._k, n)
        t = self._temperature

        # Soft membership of the top-k (long) and bottom-k (short) sets via
        # sigmoids around the k-th largest and k-th smallest score values.
        # The threshold values carry gradient to the selected entries.
        thr_long = torch.topk(scores, k, largest=True).values[-1]
        thr_short = torch.topk(scores, k, largest=False).values[-1]
        m_long = torch.sigmoid((scores - thr_long) / t)
        m_short = torch.sigmoid((thr_short - scores) / t)

        leg_long = m_long / (m_long.sum() + _EPS)
        leg_short = m_short / (m_short.sum() + _EPS)
        raw = leg_long - leg_short  # sums to zero: dollar neutral

        gross = raw.abs().sum()
        weights = raw * (self._gross / (gross + _EPS))
        return weights, portfolio_summary(weights, sigma)
