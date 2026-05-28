"""InVAR-RL-SIA Layer 2: Sparse Invariant Actor + Full-Info Critic SAC.

The SIA variant replaces the canonical SAC actor with a per-block-gated,
KL-regularised, regime-invariant bottleneck while keeping the SB3 twin-Q
critics on the FULL observation. The architectural claim (Phase 0 plan)
is that actor-critic asymmetry resolves the v3 observation-overload
pattern: the actor compresses universe-specifically; the critic always
sees the full signal.

This package exposes the re-usable building blocks plus a Stable-Baselines3
SAC subclass that wires them into the canonical InVAR-RL Layer 3
environment without changing Layer 1 or the wrapper.
"""

from invar_rl.layer2_sia.aux_loss import (
    AuxLossTerms,
    actor_aux_loss,
    actor_aux_loss_scalar,
)
from invar_rl.layer2_sia.config import SIAConfig
from invar_rl.layer2_sia.sparse_actor import (
    MLP,
    SIADims,
    SparseInvariantActor,
    resolve_dims,
)

__all__ = [
    "SIAConfig",
    "SparseInvariantActor",
    "SIADims",
    "MLP",
    "resolve_dims",
    "actor_aux_loss",
    "actor_aux_loss_scalar",
    "AuxLossTerms",
    "SACSIA",
]


def __getattr__(name):  # pragma: no cover - exercised by import only
    """Lazy import of SACSIA so the rest of the package is SB3-free.

    SB3 is GPU-heavy + brings in gym/gymnasium; tests for the actor +
    aux loss + dims should run without SB3 installed (local box).
    """
    if name == "SACSIA":
        from invar_rl.layer2_sia.sac_sia import SACSIA
        return SACSIA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
