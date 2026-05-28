"""Hyperparameter dataclass for the InVAR-RL-SIA Layer 2 SAC variant.

All numerical defaults match the Phase 0 plan; the wrapper, Layer 1 ckpts,
and the Layer 3 environment are unchanged so the SIA per-cell smoke is
comparable like-for-like against the canonical SAC tape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SIAConfig:
    """Default hyperparameters for the SIA Layer 2 SAC variant.

    Attributes:
        latent_dim: Dimension of the actor's KL-regularised latent z.
        beta_kl: Weight on the KL(q(z|input) || N(0, I)) term in the
            actor auxiliary loss.
        lambda_gate: Weight on the L1 penalty on the per-block sigmoid
            gates (encourages sparsity).
        lambda_inv: Weight on the regime-invariance penalty (variance
            across k-means-8 group means of the latent z).
        actor_hidden: Hidden-layer sizes for the actor MLPs (gate_net,
            mu_net, logvar_net, head).
        critic_hidden: Hidden-layer sizes for the SB3 twin-Q critic
            (full obs, no bottleneck).
        group_source: Source of group ids for the invariance penalty;
            currently only ``"macro_kmeans_8"`` is wired.
        total_timesteps: Total SAC env steps, matched to canonical SAC.
        learning_rate: Optimiser learning rate, shared by actor + critics.
        buffer_size: Replay buffer capacity.
        batch_size: Replay minibatch size for each training step.
        gamma: Discount factor.
        polyak_tau: Polyak averaging coefficient for the target critics.
        sparse_gates: When True (default) the actor's per-block gates are
            ``sigmoid(gate_net(obs))``. When False the gates are clamped to
            constant 1.0 for every input; the KL latent bottleneck and the
            regime-invariance penalty still fire, but the per-block sparse
            routing is disabled. Used by the Phase 4 ``no_s`` ablation.
        asymmetric_critic: When True (default) the SB3 twin-Q critic
            consumes the full observation (the canonical asymmetric AC
            setup). When False the critic is rebuilt on the actor's
            post-gate bottleneck ``actor_in`` (1 + 2 + macro_small_dim +
            4 + 1 wide); the critic now sees exactly what the actor sees.
            Used by the Phase 4 ``no_a`` ablation.
    """

    latent_dim: int = 16
    beta_kl: float = 1e-3
    lambda_gate: float = 1e-4
    lambda_inv: float = 0.1
    actor_hidden: Tuple[int, ...] = (64, 64)
    critic_hidden: Tuple[int, ...] = (128, 128)
    group_source: str = "macro_kmeans_8"
    total_timesteps: int = 20_000
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    polyak_tau: float = 5e-3
    sparse_gates: bool = True
    asymmetric_critic: bool = True
