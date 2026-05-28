"""Non-reinforcement-learning exposure baselines for Layer 3.

Three baselines, all consuming the same precomputed tape and producing a
per-step exposure multiplier:

1. Constant full exposure.
2. Volatility targeting: scale exposure to hit a fixed realised-volatility
   target.
3. Myopic supervised exposure head: a small feedforward network trained by
   supervised regression to predict a per-day separable target (the next
   day's base-book return as an information proxy) and scale exposure
   proportionally. This is the decisive baseline: beating it is what shows
   reinforcement learning is necessary rather than mere exposure scaling.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from torch import nn

from invar_rl.common.config import Layer3Config, Stage3Config
from invar_rl.layer3_control.observation import RiskState, build_observation
from invar_rl.layer3_control.precompute import EpisodeTape

_TRADING_DAYS = 252
_VOL_WINDOW = 20


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


class ExposurePolicy(Protocol):
    """A deterministic exposure rule over a tape."""

    def exposure(self, tape: EpisodeTape, t: int, risk: RiskState) -> float:
        ...


class ConstantFullExposure:
    """Always fully invested at exposure 1.0 (clipped to the bounds)."""

    def __init__(self, env_cfg: Layer3Config) -> None:
        self._lo = env_cfg.exposure_min
        self._hi = env_cfg.exposure_max

    def exposure(self, tape: EpisodeTape, t: int, risk: RiskState) -> float:
        return _clip(1.0, self._lo, self._hi)


class VolatilityTargeting:
    """Scale exposure so trailing realised vol hits an annualised target."""

    def __init__(
        self, env_cfg: Layer3Config, stage3: Stage3Config
    ) -> None:
        self._lo = env_cfg.exposure_min
        self._hi = env_cfg.exposure_max
        self._daily_target = stage3.vol_annualised_target / np.sqrt(
            _TRADING_DAYS
        )

    def exposure(self, tape: EpisodeTape, t: int, risk: RiskState) -> float:
        lo = max(0, t - _VOL_WINDOW)
        hist = tape.base_return[lo:t]
        if hist.size < 2:
            return _clip(1.0, self._lo, self._hi)
        realised = float(np.std(hist))
        if realised <= 1e-8:
            return self._hi
        return _clip(self._daily_target / realised, self._lo, self._hi)


class _MLP(nn.Module):
    """Small feedforward regressor."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MyopicExposureHead:
    """Supervised one-step exposure head (the decisive non-RL baseline).

    Trains a small MLP to regress the next day's base-book return from the
    observation, then maps the standardised prediction monotonically into
    the exposure range. It is myopic and supervised: no sequential credit
    assignment, which is exactly what the RL controller must beat.
    """

    def __init__(
        self, env_cfg: Layer3Config, stage3: Stage3Config, obs_dim: int
    ) -> None:
        self._lo = env_cfg.exposure_min
        self._hi = env_cfg.exposure_max
        self._cfg = stage3
        self._model = _MLP(obs_dim, stage3.myopic_hidden)
        self._mu = 0.0
        self._sd = 1.0

    def fit(self, tape: EpisodeTape, seed: int) -> None:
        """Train on (observation_t, base_return_{t+1}) pairs from the tape."""
        torch.manual_seed(seed)
        xs, ys = [], []
        risk = RiskState(exposure=self._lo)
        for t in range(len(tape) - 1):
            xs.append(build_observation(tape, t, risk))
            ys.append(tape.base_return[t + 1])
        x = torch.tensor(np.asarray(xs), dtype=torch.float32)
        y = torch.tensor(np.asarray(ys), dtype=torch.float32)
        opt = torch.optim.Adam(
            self._model.parameters(), lr=self._cfg.myopic_learning_rate
        )
        self._model.train()
        for _ in range(self._cfg.myopic_epochs):
            opt.zero_grad()
            loss = nn.functional.mse_loss(self._model(x), y)
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds = self._model(x).numpy()
        self._mu = float(preds.mean())
        self._sd = float(preds.std()) or 1.0

    def exposure(self, tape: EpisodeTape, t: int, risk: RiskState) -> float:
        self._model.eval()
        with torch.no_grad():
            obs = torch.tensor(
                build_observation(tape, t, risk), dtype=torch.float32
            )
            pred = float(self._model(obs.unsqueeze(0))[0])
        z = (pred - self._mu) / self._sd
        # Monotone squashing of the standardised signal into the range.
        unit = 1.0 / (1.0 + np.exp(-z))
        return _clip(
            self._lo + unit * (self._hi - self._lo), self._lo, self._hi
        )
