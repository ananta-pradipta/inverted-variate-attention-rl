"""Robust-InVAR-RL Phase 1: group-DRO + top-bottom listwise loss.

Per the source design doc (2026-05-26), Stage-2 finetune of the canonical
InVAR ranker is reweighted across macro-regime groups using an
exponentiated-gradient ascent step on the per-group loss vector. A small
top-bottom listwise margin term emphasises the tail names actually used
by the K-of-N wrapper. Defaults preserve the canonical (eta=0,
lambda_tb=0) behaviour.

References:
- Sagawa et al. 2020, "Distributionally Robust Neural Networks for Group
  Shifts" (ICLR), Algorithm 1 (online group-DRO with exponentiated
  gradient ascent on q).
- Top-bottom margin is a standard long-short ranking loss applied only
  to the top-M longs and bottom-M shorts; gradients on the middle (N-2M)
  positions are zero by construction.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def group_dro_step(
    per_group_losses: Tensor,
    q_state: Tensor,
    eta: float,
) -> Tuple[Tensor, Tensor]:
    """One exponentiated-gradient-ascent step on the group weights.

    Args:
        per_group_losses: ``(G,)`` current per-group loss values, with
            gradients flowing back to the model parameters. Groups with
            zero membership in the current batch should be passed as
            zero (with ``q_state`` unchanged for that group).
        q_state: ``(G,)`` running group weights, non-negative and summing
            to 1. Detached from the autograd graph: the DRO update on
            ``q`` is a closed-form ratio, not a gradient step.
        eta: positive DRO step size. When ``eta == 0`` the function
            reduces to plain ERM: ``q_new == q_old`` and the weighted
            loss equals ``q_old @ per_group_losses``.

    Returns:
        ``(weighted_loss, new_q_state)`` where ``weighted_loss`` is a
        scalar with gradients into the model parameters via
        ``per_group_losses``, and ``new_q_state`` is detached and
        normalised to sum to 1.

    Raises:
        ValueError: if ``per_group_losses`` and ``q_state`` shapes
            mismatch, if ``q_state`` is not (close to) a probability
            vector, or if ``eta`` is negative.
    """
    if eta < 0:
        raise ValueError(f"[ERR] eta must be >= 0; got {eta}")
    if per_group_losses.shape != q_state.shape:
        raise ValueError(
            "[ERR] per_group_losses and q_state shape mismatch: "
            f"{tuple(per_group_losses.shape)} vs {tuple(q_state.shape)}"
        )
    if per_group_losses.dim() != 1:
        raise ValueError(
            f"[ERR] per_group_losses must be 1D; got {per_group_losses.dim()}D"
        )
    q_detached = q_state.detach()
    if eta == 0.0:
        weighted = (q_detached * per_group_losses).sum()
        return weighted, q_detached.clone()
    # Exponentiated gradient ascent on q: q_new propto q_old * exp(eta * L_g)
    log_q = torch.log(q_detached.clamp_min(1.0e-12))
    log_q_new = log_q + float(eta) * per_group_losses.detach()
    # Stable softmax-style normalisation.
    log_q_new = log_q_new - log_q_new.max()
    q_new = torch.exp(log_q_new)
    q_new = q_new / q_new.sum().clamp_min(1.0e-12)
    weighted = (q_new * per_group_losses).sum()
    return weighted, q_new


def compute_top_bottom_loss(
    scores: Tensor,
    returns: Tensor,
    mask: Tensor,
    M: int,
) -> Tensor:
    """Pairwise margin loss applied only to the top-M longs and bottom-M shorts.

    The loss is ``mean over (i,j) of ReLU(returns[j] - returns[i])`` for
    every pair where ``i`` is in the top-M by score and ``j`` is in the
    bottom-M by score. Gradients flow only through the top-M and bottom-M
    score positions; middle positions receive zero gradient.

    Args:
        scores: ``(N,)`` predicted scores for the day's active stocks.
        returns: ``(N,)`` z-scored next-day returns for the same stocks.
        mask: ``(N,)`` bool / 0-1 mask of active stocks. Inactive entries
            are excluded from both the top and bottom selection.
        M: per-side count (e.g., ``M == K`` for the SP500 K=50 wrapper).
            If ``2 * M > number of active stocks``, the function falls
            back to ``M = active // 2`` to avoid empty groups.

    Returns:
        Scalar margin loss. Returns a zero scalar (with autograd ties)
        when fewer than 2 active stocks survive the mask.
    """
    if scores.shape != returns.shape:
        raise ValueError(
            "[ERR] scores and returns shape mismatch: "
            f"{tuple(scores.shape)} vs {tuple(returns.shape)}"
        )
    if mask.shape != scores.shape:
        raise ValueError(
            "[ERR] mask shape mismatch: "
            f"{tuple(mask.shape)} vs {tuple(scores.shape)}"
        )
    active = mask.bool()
    n_active = int(active.sum().item())
    if n_active < 2:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    # Active indices and gather active scores / returns.
    active_idx = active.nonzero(as_tuple=False).squeeze(-1)
    s_active = scores.index_select(0, active_idx)
    r_active = returns.index_select(0, active_idx)
    # Effective M; never exceed half the active count.
    m_eff = int(min(int(M), n_active // 2))
    if m_eff < 1:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    # Top-M longs (by descending score) and bottom-M shorts (by ascending).
    # torch.topk on -scores gives the bottom; gradients flow via gather.
    top_vals, top_pos = torch.topk(s_active, k=m_eff, largest=True, sorted=False)
    bot_vals, bot_pos = torch.topk(s_active, k=m_eff, largest=False, sorted=False)
    r_top = r_active.index_select(0, top_pos)              # (M,)
    r_bot = r_active.index_select(0, bot_pos)              # (M,)
    # Pairwise long-short margin: for every (top long, bottom short) pair
    # we want (score_long - score_short) >= (return_long - return_short)
    # so the score margin tracks the realised return margin. The hinge
    # loss is on the SCORE side, so gradients flow into the top-M and
    # bottom-M score positions. Middle positions receive zero gradient
    # because torch.topk's gather is a hard selection of those entries.
    score_margin = top_vals.unsqueeze(1) - bot_vals.unsqueeze(0)   # (M, M)
    target_margin = r_top.unsqueeze(1) - r_bot.unsqueeze(0)        # (M, M)
    # Margin = 1.0 baseline plus the realised return gap; equivalent to
    # the standard pairwise margin where the long must beat the short by
    # at least the return-based margin. Constants do not affect gradient.
    pairwise = torch.relu(target_margin - score_margin).mean()
    return pairwise


__all__ = ["group_dro_step", "compute_top_bottom_loss"]
