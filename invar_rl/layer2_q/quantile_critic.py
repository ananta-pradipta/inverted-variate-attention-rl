"""Quantile critic for InVAR-RL-Q.

Each critic is an MLP that takes the concatenated ``[obs, action]`` vector
and outputs ``n_quantiles`` predicted quantile values of the return-to-go
distribution. The SACQ subclass instantiates a twin pair (Q1, Q2) for
clipped double-Q at the quantile level. The Bellman target is computed in
the SACQ.train() loop; the huber-quantile loss helper below is reused for
both critics.

The quantile midpoints are tau_i = (i - 0.5) / Nq for i in 1..Nq. With
Nq = 51 the midpoints are 0.0098, 0.0294, ..., 0.9902. The standard QR-DQN
loss is the elementwise asymmetric huber loss weighted by
``|tau - I(td < 0)|`` and reduced as mean over both axes.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _make_mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
) -> nn.Sequential:
    """Build a plain ReLU MLP; output is linear (no activation)."""
    layers = []
    prev = int(in_dim)
    for h in hidden:
        layers.append(nn.Linear(prev, int(h)))
        layers.append(nn.ReLU(inplace=False))
        prev = int(h)
    layers.append(nn.Linear(prev, int(out_dim)))
    return nn.Sequential(*layers)


class QuantileCritic(nn.Module):
    """Twin quantile critic over Nq quantile midpoints of Q(s, a).

    The module holds n_critics independent quantile heads (n_critics=2 by
    default to mirror SB3's clipped double-Q). Each head maps
    ``[obs, action]`` to an ``Nq``-dim quantile vector. The forward
    returns a tuple of n_critics tensors, one per head, each of shape
    ``[B, Nq]``. This mirrors SB3 ``ContinuousCritic.forward`` which
    returns a tuple of n_critics tensors of shape ``[B, 1]``; the only
    change is the per-tensor output dim.

    Args:
        obs_dim: Observation dimension (post-flatten).
        action_dim: Action dimension. SACQ uses 1 (scalar exposure).
        n_quantiles: Number of quantile midpoints per critic. Default 51.
        hidden: Hidden-layer sizes for each critic MLP. Default [256, 256].
        n_critics: Number of critic heads; SB3 SAC uses 2 (clipped double-Q).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 1,
        n_quantiles: int = 51,
        hidden: Sequence[int] = (256, 256),
        n_critics: int = 2,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if n_quantiles <= 0:
            raise ValueError("n_quantiles must be positive")
        if n_critics <= 0:
            raise ValueError("n_critics must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_quantiles = int(n_quantiles)
        self.n_critics = int(n_critics)
        self.q_networks = nn.ModuleList(
            [
                _make_mlp(
                    in_dim=self.obs_dim + self.action_dim,
                    hidden=hidden,
                    out_dim=self.n_quantiles,
                )
                for _ in range(self.n_critics)
            ]
        )
        # Cached quantile midpoints tau in (0, 1); registered as a buffer
        # so it tracks the module device under .to(device) and is included
        # in state_dict for reproducibility / save / load.
        taus = (torch.arange(self.n_quantiles).float() + 0.5) / float(
            self.n_quantiles
        )
        self.register_buffer("taus", taus.view(1, self.n_quantiles))

    def set_training_mode(self, mode: bool) -> None:
        """SB3 BaseModel-compatibility shim: toggle train/eval mode."""
        self.train(mode)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple:
        """Return one ``[B, Nq]`` quantile tensor per critic head.

        Args:
            obs: ``[B, obs_dim]`` float tensor.
            action: ``[B, action_dim]`` float tensor.

        Returns:
            Tuple of length ``n_critics``; each element is a ``[B, Nq]``
            tensor of predicted quantile values for that head.
        """
        if obs.dim() != 2:
            obs = obs.reshape(obs.shape[0], -1)
        if action.dim() != 2:
            action = action.reshape(action.shape[0], -1)
        x = torch.cat([obs, action], dim=-1)
        return tuple(q_net(x) for q_net in self.q_networks)


def huber_quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Asymmetric huber-quantile loss for a single critic head.

    Implements the standard QR-DQN / IQN quantile regression loss with a
    huber smoothing constant kappa=1.0. The per-element loss is

        rho_tau(td) = |tau - I(td < 0)| * huber(td, kappa) / kappa

    where ``td = target_j - pred_i`` is the TD error between the j-th
    target quantile and the i-th predicted quantile. The loss is reduced
    as a mean over both the target-quantile axis (j) and the predicted-
    quantile axis (i), then averaged over the batch axis. This matches
    Dabney et al. 2018 eq. 10.

    Args:
        pred: ``[B, Nq]`` predicted quantile tensor.
        target: ``[B, Nq]`` target quantile tensor (no grad).
        taus: ``[1, Nq]`` quantile midpoints (matches pred's quantile axis).
        kappa: Huber smoothing constant; 1.0 matches the canonical IQN /
            QR-DQN setting.

    Returns:
        Scalar tensor; the mean huber-quantile loss for this batch.
    """
    if pred.dim() != 2:
        raise ValueError(f"pred must be [B, Nq]; got {tuple(pred.shape)}")
    if target.dim() != 2:
        raise ValueError(f"target must be [B, Nq]; got {tuple(target.shape)}")
    if pred.shape[0] != target.shape[0]:
        raise ValueError(
            "pred and target must share the batch axis; got "
            f"{pred.shape[0]} vs {target.shape[0]}"
        )
    n_q_pred = pred.shape[1]
    n_q_target = target.shape[1]
    # td[b, i, j] = target[b, j] - pred[b, i]; shape [B, Nq_pred, Nq_target]
    td = target.unsqueeze(1) - pred.unsqueeze(2)
    # Huber loss in TD-error space; smoothing region is |td| < kappa.
    abs_td = td.abs()
    is_smooth = (abs_td <= kappa).float()
    huber = is_smooth * 0.5 * td.pow(2) + (1.0 - is_smooth) * kappa * (
        abs_td - 0.5 * kappa
    )
    # Weighting: |tau - I(td < 0)|. taus has shape [1, Nq_pred]; broadcast
    # along the target-quantile axis.
    if taus.dim() != 2 or taus.shape[1] != n_q_pred:
        raise ValueError(
            f"taus must be shape [1, Nq_pred]; got {tuple(taus.shape)} "
            f"for Nq_pred={n_q_pred}"
        )
    taus_b = taus.unsqueeze(2)  # [1, Nq_pred, 1]
    weight = (taus_b - (td.detach() < 0.0).float()).abs()
    loss = weight * huber / max(kappa, 1e-8)
    # Reduce: mean over the target-quantile axis (j), sum over the
    # predicted-quantile axis (i) is the original Dabney formulation;
    # IQN / SB3-DQN-extensions take a mean over both axes for stability
    # at very different Nq settings. We use mean-over-both to keep the
    # gradient magnitude Nq-invariant.
    return loss.mean()
