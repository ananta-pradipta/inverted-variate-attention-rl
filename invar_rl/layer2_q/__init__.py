"""InVAR-RL-Q Layer 2: Quantile-Distributional SAC.

The Q variant replaces SB3 SAC's twin scalar Q-critics with twin quantile
critics over the return distribution and replaces the actor objective
with a mean / CVaR blend. The actor architecture, the wrapper, the Layer
1 ckpts, and the Layer 3 environment are unchanged so the Q-vs-canonical
delta isolates the critic + actor objective change.

Mechanism claim (Phase 0): scalar mean-style critics are mildly mis-
aligned with the regime-stress objective; replacing them with quantile
critics + CVaR-blend actor should lift SP500 F2 (the left-tail-poor
fold) without sacrificing pooled Sharpe.

This package exposes the QuantileCritic + CVaR helper + SACQ subclass.
The SB3 SAC subclass is lazy-imported so the rest of the package is
SB3-free (mirroring the layer2_sia / layer2_ur convention).
"""

from invar_rl.layer2_q.config import QConfig
from invar_rl.layer2_q.cvar import cvar_from_quantiles
from invar_rl.layer2_q.quantile_critic import (
    QuantileCritic,
    huber_quantile_loss,
)

__all__ = [
    "QConfig",
    "QuantileCritic",
    "huber_quantile_loss",
    "cvar_from_quantiles",
    "SACQ",
]


def __getattr__(name):  # pragma: no cover - exercised by import only
    """Lazy import of SACQ so the rest of the package is SB3-free."""
    if name == "SACQ":
        from invar_rl.layer2_q.sac_q import SACQ
        return SACQ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
