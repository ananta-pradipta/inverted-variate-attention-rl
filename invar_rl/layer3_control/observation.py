"""Layer 3 observation construction.

The observation is built only from precomputed, detached Layer 1 and Layer 2
values plus the agent's own risk state. The true latent regime is never
included. The exact layout is fixed and documented below so the policy and
value networks have a stable interface.

Layout (length = 7 + macro_dim), all values known as of step ``t``:

    index 0                : Layer 1 cross-sectional score dispersion
    index 1                : Layer 2 predicted portfolio volatility
    index 2                : Layer 2 effective number of positions
    index 3                : rolling realised volatility of the strategy
    index 4                : current drawdown from the high-water mark
    index 5                : current exposure
    index 6                : days since the last detected regime change
    index 7 .. 7+macro_dim : Layer 1 macro-regime encoding
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from invar_rl.layer3_control.precompute import EpisodeTape

_FIXED_FIELDS = 7


def observation_dim(tape: EpisodeTape) -> int:
    """Total observation length for a given tape."""
    return _FIXED_FIELDS + tape.macro_dim


@dataclass
class RiskState:
    """Mutable agent risk state, updated each step from realised outcomes."""

    rolling_vol: float = 0.0
    drawdown: float = 0.0
    exposure: float = 0.0
    days_since_regime_change: float = 0.0


def build_observation(
    tape: EpisodeTape, t: int, risk: RiskState
) -> np.ndarray:
    """Assemble the observation vector for step ``t``.

    Args:
        tape: The precomputed episode tape.
        t: Step position within the tape (0-based). Only indices at or
            before ``t`` are read, so the observation carries no future
            information.
        risk: The agent's current risk state.

    Returns:
        A float32 vector of length ``observation_dim(tape)``.
    """
    fixed = np.array(
        [
            tape.score_dispersion[t],
            tape.pred_vol[t],
            tape.eff_positions[t],
            risk.rolling_vol,
            risk.drawdown,
            risk.exposure,
            risk.days_since_regime_change,
        ],
        dtype=np.float32,
    )
    enc = tape.macro_encoding[t].astype(np.float32)
    return np.concatenate([fixed, enc])
