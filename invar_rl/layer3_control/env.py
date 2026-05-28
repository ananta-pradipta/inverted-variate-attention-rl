"""Gymnasium environment for the Layer 3 exposure controller.

Layers 1 and 2 are frozen and precomputed (see ``precompute``). The
environment replays the detached tape: the action scales the precomputed
base book, the realised strategy return follows from the precomputed base
return, and the risk state and reward are updated. There is no gradient
connection to the lower layers.

Two environments are provided. ``ExposureEnv`` is the main exact-replay
environment. ``ExposureEnvBootstrapResiduals`` perturbs the base-return
stream with volatility-stratified bootstrapped residuals and is for
robustness testing only; it is deliberately a separate class so it is never
confused with exact replay. Neither environment ever exposes the true latent
regime.
"""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from invar_rl.common.config import Layer3Config
from invar_rl.layer3_control.observation import (
    RiskState,
    build_observation,
    observation_dim,
)
from invar_rl.layer3_control.precompute import EpisodeTape
from invar_rl.layer3_control.reward import RewardFunction

_VOL_WINDOW = 20
_REGIME_Z = 3.0


class ExposureEnv(gym.Env):
    """Exact-replay exposure-control environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        tape: EpisodeTape,
        cfg: Layer3Config,
        bootstrap_episode: bool = False,
    ) -> None:
        """Initialise the environment.

        Args:
            tape: Precomputed frozen Layer 1 and Layer 2 outputs.
            cfg: Environment and reward configuration.
            bootstrap_episode: If True, each reset samples a random
                contiguous episode window of length ``cfg.episode_days``
                within the tape; otherwise the full tape is one episode.
        """
        super().__init__()
        if len(tape) < 2:
            raise ValueError("tape must contain at least two steps")
        self._tape = tape
        self._cfg = cfg
        self._bootstrap = bootstrap_episode
        self._ep_len = min(cfg.episode_days, len(tape))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim(tape),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.float32(cfg.exposure_min),
            high=np.float32(cfg.exposure_max),
            shape=(1,),
            dtype=np.float32,
        )
        self._reward_fn = RewardFunction(cfg)
        self._risk = RiskState()
        self._start = 0
        self._t = 0
        self._equity = 1.0
        self._hwm = 1.0
        self._ret_hist: list[float] = []
        self._pvol_hist: list[float] = []

    # Episode return stream; overridden by the bootstrap variant.
    def _episode_base_return(self) -> np.ndarray:
        return self._tape.base_return

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        max_start = len(self._tape) - self._ep_len
        if self._bootstrap and max_start > 0:
            self._start = int(self.np_random.integers(0, max_start + 1))
        else:
            self._start = 0
        self._t = 0
        self._equity = 1.0
        self._hwm = 1.0
        self._ret_hist = []
        self._pvol_hist = []
        self._reward_fn.reset()
        self._risk = RiskState(
            rolling_vol=0.0,
            drawdown=0.0,
            exposure=float(self._cfg.exposure_min),
            days_since_regime_change=0.0,
        )
        obs = build_observation(self._tape, self._start, self._risk)
        return obs.astype(np.float32), {}

    def _detect_regime_change(self, pvol: float) -> None:
        """Increment the regime-change counter; reset on a volatility jump."""
        self._pvol_hist.append(pvol)
        if len(self._pvol_hist) > _VOL_WINDOW:
            ref = np.asarray(self._pvol_hist[-(_VOL_WINDOW + 1):-1])
            mu, sd = ref.mean(), ref.std()
            if sd > 1e-12 and abs(pvol - mu) > _REGIME_Z * sd:
                self._risk.days_since_regime_change = 0.0
                return
        self._risk.days_since_regime_change += 1.0

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        idx = self._start + self._t
        base_ret = self._episode_base_return()

        raw = float(np.asarray(action).reshape(-1)[0])
        target = float(
            np.clip(raw, self._cfg.exposure_min, self._cfg.exposure_max)
        )
        prev = self._risk.exposure
        band = self._cfg.exposure_change_band
        exposure = float(np.clip(target, prev - band, prev + band))
        exposure = float(
            np.clip(exposure, self._cfg.exposure_min, self._cfg.exposure_max)
        )

        strat_ret = exposure * float(base_ret[idx])
        traded_notional = abs(exposure - prev) * float(
            self._tape.base_gross[idx]
        )

        self._equity *= 1.0 + strat_ret
        self._hwm = max(self._hwm, self._equity)
        self._risk.drawdown = (
            0.0 if self._hwm <= 0 else 1.0 - self._equity / self._hwm
        )
        self._ret_hist.append(strat_ret)
        if len(self._ret_hist) >= 2:
            self._risk.rolling_vol = float(
                np.std(self._ret_hist[-_VOL_WINDOW:])
            )
        d_exposure = exposure - prev
        self._risk.exposure = exposure
        self._detect_regime_change(float(self._tape.pred_vol[idx]))

        reward = self._reward_fn(
            strategy_return=strat_ret,
            drawdown=self._risk.drawdown,
            delta_exposure=d_exposure,
            traded_notional=traded_notional,
        )

        self._t += 1
        terminated = False
        truncated = self._t >= self._ep_len - 1
        obs_idx = min(self._start + self._t, len(self._tape) - 1)
        obs = build_observation(self._tape, obs_idx, self._risk)
        info = {
            "equity": self._equity,
            "exposure": exposure,
            "strategy_return": strat_ret,
        }
        return obs.astype(np.float32), reward, terminated, truncated, info


class ExposureEnvBootstrapResiduals(ExposureEnv):
    """Robustness-only variant: volatility-stratified residual bootstrap.

    The base-return stream is decomposed into a rolling mean plus residuals.
    On each reset the residuals are resampled within volatility-bucket strata
    (a proxy that never reads the true latent regime) and the stream is
    reconstructed. This is for robustness testing only and is never used for
    exact-replay evaluation.
    """

    def __init__(
        self,
        tape: EpisodeTape,
        cfg: Layer3Config,
        bootstrap_episode: bool = False,
        n_strata: int = 3,
    ) -> None:
        super().__init__(tape, cfg, bootstrap_episode=bootstrap_episode)
        self._n_strata = max(1, int(n_strata))
        self._perturbed = self._tape.base_return.copy()

    def _episode_base_return(self) -> np.ndarray:
        return self._perturbed

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        obs, info = super().reset(seed=seed, options=options)
        base = self._tape.base_return
        w = min(_VOL_WINDOW, max(2, base.shape[0] // 10))
        roll = np.convolve(
            base, np.ones(w) / w, mode="same"
        )
        resid = base - roll

        # Strata by realised-volatility bucket of the predicted-vol series,
        # a regime-agnostic proxy. Resample residuals within each stratum.
        ranks = np.argsort(np.argsort(self._tape.pred_vol))
        strata = (ranks * self._n_strata) // max(1, len(ranks))
        perturbed = roll.copy()
        for s in range(self._n_strata):
            mask = strata == s
            if mask.sum() > 0:
                pool = resid[mask]
                draws = self.np_random.integers(0, pool.shape[0], mask.sum())
                perturbed[mask] = roll[mask] + pool[draws]
        self._perturbed = perturbed
        return obs, info
