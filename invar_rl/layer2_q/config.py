"""Hyperparameter dataclass for the InVAR-RL-Q Layer 2 SAC variant.

All numerical defaults match the Phase 0 plan; the wrapper, Layer 1 ckpts,
and the Layer 3 environment are unchanged so the Q per-cell smoke is
comparable like-for-like against the canonical SAC tape and the SIA tape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class QConfig:
    """Default hyperparameters for the Quantile-distributional Layer 2 SAC.

    Attributes:
        n_quantiles: Number of quantile midpoints estimated by each critic.
            Quantile midpoints are tau_i = (i - 0.5) / Nq for i in 1..Nq.
        alpha_cvar: Lower-tail level for the CVaR statistic in the actor
            objective; 0.1 means "average of the lower-decile quantiles".
        eta_blend: Mean / CVaR blend coefficient for the actor objective;
            actor target = eta * q_mean + (1 - eta) * q_cvar. eta = 1.0
            recovers the canonical SAC mean-Q actor target; eta = 0.0 is
            pure CVaR.
        actor_hidden: Hidden-layer sizes for the SB3 SAC actor MLP.
            Defaults match the SIA actor_hidden (64, 64); the canonical
            SAC default is [256, 256], but the Phase 0 plan keeps the
            actor narrow to isolate the critic-side change.
        critic_hidden: Hidden-layer sizes for the QuantileCritic MLPs.
        total_timesteps: Total SAC env steps, matched to canonical SAC.
        learning_rate: Optimiser learning rate shared by actor + critics.
        buffer_size: Replay buffer capacity.
        batch_size: Replay minibatch size for each gradient step.
        gamma: Discount factor.
        polyak_tau: Polyak averaging coefficient for the target critics.
    """

    n_quantiles: int = 51
    alpha_cvar: float = 0.1
    eta_blend: float = 0.5
    actor_hidden: List[int] = field(default_factory=lambda: [64, 64])
    critic_hidden: List[int] = field(default_factory=lambda: [256, 256])
    total_timesteps: int = 20_000
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    polyak_tau: float = 5e-3
