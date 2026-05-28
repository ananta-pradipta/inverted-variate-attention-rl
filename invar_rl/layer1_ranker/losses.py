"""Cross-sectional ranking losses.

The default is the cross-sectional mean squared error between predicted
scores and within-day z-scored forward returns, computed only over the
tradable and labelled stocks. A listwise ranking loss is provided as a
configurable alternative.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def _select(
    scores: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict to tradable, labelled, finite entries."""
    valid = mask & torch.isfinite(target)
    return scores[valid], target[valid]


def cross_sectional_mse(
    scores: torch.Tensor, target_z: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over the within-day tradable, labelled set.

    Args:
        scores: Predicted scores, shape (n_stocks,).
        target_z: Within-day z-scored forward returns, shape (n_stocks,).
        mask: Boolean tradable-and-labelled mask, shape (n_stocks,).

    Returns:
        A scalar loss. Returns zero (with grad) if fewer than two valid
        stocks are present on the day.
    """
    s, t = _select(scores, target_z, mask)
    if s.numel() < 2:
        return scores.sum() * 0.0
    return F.mse_loss(s, t)


def listwise_rank_loss(
    scores: torch.Tensor, target_z: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """ListNet top-one listwise cross-entropy.

    Compares the softmax distribution over predicted scores with the softmax
    distribution over the z-scored targets across the day's cross-section.

    Args:
        scores: Predicted scores, shape (n_stocks,).
        target_z: Within-day z-scored forward returns, shape (n_stocks,).
        mask: Boolean tradable-and-labelled mask, shape (n_stocks,).

    Returns:
        A scalar loss. Returns zero (with grad) if fewer than two valid
        stocks are present on the day.
    """
    s, t = _select(scores, target_z, mask)
    if s.numel() < 2:
        return scores.sum() * 0.0
    target_p = torch.softmax(t, dim=0)
    pred_log_p = torch.log_softmax(s, dim=0)
    return -(target_p * pred_log_p).sum()


_LOSSES = {
    "cross_sectional_mse": cross_sectional_mse,
    "listwise_rank": listwise_rank_loss,
}


def get_loss(
    kind: str,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the loss callable named by ``kind``.

    Args:
        kind: Either "cross_sectional_mse" or "listwise_rank".

    Returns:
        The loss function.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind not in _LOSSES:
        raise ValueError(
            f"unknown loss kind {kind!r}, expected one of {sorted(_LOSSES)}"
        )
    return _LOSSES[kind]
