"""Observation-mask wrapper for the layer-3 stripped-observation ablation.

Zeros out the Layer 1 and Layer 2 fields of the observation (score
dispersion, predicted vol, effective positions, macro encoding) so
the RL agent only sees its own risk state (rolling vol, drawdown,
current exposure, days since regime change). The action and reward
mechanics are unchanged; the agent simply cannot condition on what
Layer 1 + Layer 2 produced today.

Ablation 3 in the InVAR-RL ablation suite. If the agent's pooled
Sharpe collapses under this wrapper, it confirms the Layer 3
contribution is driven by reading the Layer 1 + 2 regime signal,
not by the agent's own bookkeeping.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


_FIXED_FIELDS = 7
_L1_L2_FIXED_IDX = (0, 1, 2)
_RISK_KEEP_IDX = (3, 4, 5, 6)


class StrippedObservationWrapper(gym.ObservationWrapper):
    """Mask the Layer-1 / Layer-2 fields of an ExposureEnv observation.

    Observation layout (from :mod:`invar_rl.layer3_control.observation`):

    ``[score_dispersion, pred_vol, eff_positions,
       rolling_vol, drawdown, exposure, days_since_regime_change,
       macro_encoding...]``

    The wrapper zeros indices 0, 1, 2 and the entire macro_encoding tail.
    Risk-state fields (3-6) are preserved.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._n = int(env.observation_space.shape[0])

    def observation(self, obs: np.ndarray) -> np.ndarray:
        masked = np.array(obs, dtype=np.float32, copy=True)
        for i in _L1_L2_FIXED_IDX:
            masked[i] = 0.0
        if self._n > _FIXED_FIELDS:
            masked[_FIXED_FIELDS:] = 0.0
        return masked


__all__ = ["StrippedObservationWrapper"]
