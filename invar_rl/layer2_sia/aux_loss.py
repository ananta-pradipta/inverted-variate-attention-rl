"""Auxiliary actor loss for InVAR-RL-SIA.

Three additive terms, each weighted:

  1. ``kl``: KL divergence from N(mu, exp(logvar)) to the standard normal,
     summed over the latent dim, mean over the batch. Regularises the
     stochastic bottleneck.
  2. ``gate_l1``: mean L1 magnitude of the per-block sigmoid gates,
     encouraging the actor to close gates on uninformative blocks.
  3. ``inv``: regime-invariance penalty; computed as the variance across
     group means of the latent mean z = mu, where groups are the k-means-8
     macro clusters per training day (see
     :mod:`invar_rl.layer2_sia.regime_probs`). Encourages the latent code
     to look similar across macro regimes seen in the same minibatch, so
     the agent's policy generalises across regimes the L1 ranker has not
     seen.

Total auxiliary loss::

    L_aux = beta_kl * kl + lambda_gate * gate_l1 + lambda_inv * inv

This is ADDED to the standard SAC actor loss inside
:class:`invar_rl.layer2_sia.sac_sia.SACSIA.train`. The standard SAC actor
loss uses the SB3 twin-Q critics on the FULL observation (no bottleneck);
the actor itself sees only the gated, bottlenecked latent z.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class AuxLossTerms:
    """Per-term aux-loss values exposed for logging.

    All scalars are detached so the dataclass can be moved off-device
    safely; the gradient still flows through the live total returned by
    :func:`actor_aux_loss` (the returned scalar is the differentiable one).
    """

    kl: float
    gate_l1: float
    inv: float
    total: float


def _kl_to_standard_normal(
    mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, I)), summed over latent, mean over batch.

    Closed form: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) per row,
    then mean across rows.
    """
    if mu.dim() != 2 or logvar.dim() != 2:
        raise ValueError(
            f"mu and logvar must be (B, D); got {tuple(mu.shape)} "
            f"and {tuple(logvar.shape)}"
        )
    if mu.shape != logvar.shape:
        raise ValueError(
            f"mu and logvar must match; got {tuple(mu.shape)} vs "
            f"{tuple(logvar.shape)}"
        )
    per_row = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
    return per_row.mean()


def _gate_l1(gates: torch.Tensor) -> torch.Tensor:
    """Mean absolute value of the sigmoid gates.

    Because the gates pass through a sigmoid they are already in [0, 1];
    the absolute value reduces to the gate value itself, but we keep the
    abs() so the formula is faithful to the Phase 0 spec and so any future
    relaxation of the gate range still yields a non-negative penalty.
    """
    if gates.dim() != 2:
        raise ValueError(
            f"gates must be (B, K); got {tuple(gates.shape)}"
        )
    return gates.abs().mean()


def _regime_invariance(
    mu: torch.Tensor, group_ids: torch.Tensor
) -> torch.Tensor:
    """Variance across group means of the latent mu.

    For each unique ``g`` in ``group_ids``, compute the mean of mu over
    rows where group_ids == g. Stack the group means (G, D) and return
    the elementwise variance across the group axis, mean-reduced over D.

    When all rows share a single group, returns zero (no invariance signal
    available in this minibatch).
    """
    if mu.dim() != 2:
        raise ValueError(f"mu must be (B, D); got {tuple(mu.shape)}")
    if group_ids.dim() != 1:
        raise ValueError(
            f"group_ids must be 1-D; got {tuple(group_ids.shape)}"
        )
    if group_ids.shape[0] != mu.shape[0]:
        raise ValueError(
            f"group_ids length {group_ids.shape[0]} != mu batch "
            f"{mu.shape[0]}"
        )
    unique_groups = torch.unique(group_ids)
    if unique_groups.numel() <= 1:
        return torch.zeros((), dtype=mu.dtype, device=mu.device)
    group_means = []
    for g in unique_groups:
        mask = (group_ids == g)
        group_means.append(mu[mask].mean(dim=0))
    stacked = torch.stack(group_means, dim=0)  # (G, D)
    # Variance across G, mean over D. Unbiased=False matches the Phase 0
    # spec "((group_means - mean)^2).mean()" which is the population variance
    # over G times D, equivalent to mean of per-D population variance.
    centered = stacked - stacked.mean(dim=0, keepdim=True)
    return (centered.pow(2)).mean()


def actor_aux_loss(
    aux: Dict[str, torch.Tensor],
    group_ids: Optional[torch.Tensor],
    beta_kl: float,
    lambda_gate: float,
    lambda_inv: float,
) -> AuxLossTerms:
    """Compute the three SIA auxiliary loss terms and their weighted sum.

    Args:
        aux: Dict with keys ``mu``, ``logvar``, ``gates`` (the second
            return of :class:`SparseInvariantActor.forward`).
        group_ids: 1-D long tensor of length B carrying the k-means-8 hard
            assignment per row. If None, the invariance term is zero
            (useful for smoke runs when the regime cache is missing).
        beta_kl: Weight on the KL term.
        lambda_gate: Weight on the gate L1 term.
        lambda_inv: Weight on the regime-invariance term.

    Returns:
        :class:`AuxLossTerms` holding (kl, gate_l1, inv, total). The total
        value carries gradients; the per-term floats are detached.
    """
    if "mu" not in aux or "logvar" not in aux or "gates" not in aux:
        raise KeyError(
            "aux must contain mu, logvar, gates; got keys " + str(list(aux))
        )
    kl_t = _kl_to_standard_normal(aux["mu"], aux["logvar"])
    gate_t = _gate_l1(aux["gates"])
    if group_ids is None:
        inv_t = torch.zeros((), dtype=aux["mu"].dtype, device=aux["mu"].device)
    else:
        inv_t = _regime_invariance(aux["mu"], group_ids)
    total = (
        float(beta_kl) * kl_t
        + float(lambda_gate) * gate_t
        + float(lambda_inv) * inv_t
    )
    return AuxLossTerms(
        kl=float(kl_t.detach().item()),
        gate_l1=float(gate_t.detach().item()),
        inv=float(inv_t.detach().item()),
        total=float(total.detach().item()),
    ), total


def actor_aux_loss_scalar(
    aux: Dict[str, torch.Tensor],
    group_ids: Optional[torch.Tensor],
    beta_kl: float,
    lambda_gate: float,
    lambda_inv: float,
) -> torch.Tensor:
    """Convenience wrapper returning only the differentiable total scalar."""
    _, total = actor_aux_loss(
        aux=aux,
        group_ids=group_ids,
        beta_kl=beta_kl,
        lambda_gate=lambda_gate,
        lambda_inv=lambda_inv,
    )
    return total
