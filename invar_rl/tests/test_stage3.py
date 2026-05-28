"""Tests for Stage 3: RL controller construction and the non-RL baselines.

These are lightweight soundness checks on a fabricated tape (no frozen
lower stack required). The full train-to-completion acceptance is the
Wulver Stage 3 sweep, not a unit test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from invar_rl.common.config import (
    load_layer3_config,
    load_stage3_config,
)
from invar_rl.common.seeding import make_rng
from invar_rl.baselines.exposure_baselines import (
    ConstantFullExposure,
    MyopicExposureHead,
    VolatilityTargeting,
)
from invar_rl.layer3_control.agent import build_agent
from invar_rl.layer3_control.env import ExposureEnv
from invar_rl.layer3_control.observation import RiskState
from invar_rl.layer3_control.precompute import EpisodeTape

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _tape(t: int = 120, d: int = 6, seed: int = 42) -> EpisodeTape:
    rng = make_rng(seed)
    return EpisodeTape(
        days=np.arange(t, dtype=np.int64),
        score_dispersion=np.abs(rng.normal(1.0, 0.2, t)),
        macro_encoding=rng.normal(0.0, 1.0, size=(t, d)),
        pred_vol=np.abs(rng.normal(0.1, 0.02, t)),
        eff_positions=rng.uniform(5.0, 40.0, t),
        base_return=rng.normal(0.0008, 0.01, t),
        base_gross=np.ones(t),
        daily_ic=rng.normal(0.02, 0.05, t),
    )


def _cfgs():
    return (
        load_layer3_config(CONFIG_DIR / "layer3.yaml"),
        load_stage3_config(CONFIG_DIR / "stage3.yaml"),
    )


def test_non_rl_baselines_emit_in_range_exposure() -> None:
    env_cfg, stage3 = _cfgs()
    tape = _tape()
    risk = RiskState(exposure=env_cfg.exposure_min)
    policies = [
        ConstantFullExposure(env_cfg),
        VolatilityTargeting(env_cfg, stage3),
    ]
    myopic = MyopicExposureHead(
        env_cfg, stage3, obs_dim=7 + tape.macro_dim
    )
    myopic.fit(tape, seed=42)
    policies.append(myopic)
    for pol in policies:
        for t in (5, 50, 100):
            e = pol.exposure(tape, t, risk)
            assert env_cfg.exposure_min <= e <= env_cfg.exposure_max


def test_recurrent_ppo_trains_briefly_and_predicts_in_range() -> None:
    env_cfg, stage3 = _cfgs()
    env = ExposureEnv(_tape(), env_cfg)
    agent = build_agent("recurrent_ppo", env, stage3, seed=42)
    agent.learn(total_timesteps=64)
    obs, _ = env.reset(seed=0)
    action, _ = agent.predict(
        obs, state=None, episode_start=np.ones((1,), dtype=bool),
        deterministic=True,
    )
    a = float(np.asarray(action).reshape(-1)[0])
    assert env_cfg.exposure_min - 1e-5 <= a <= env_cfg.exposure_max + 1e-5


def test_feedforward_ppo_and_sac_build_and_step() -> None:
    env_cfg, stage3 = _cfgs()
    for method in ("feedforward_ppo", "sac"):
        env = ExposureEnv(_tape(), env_cfg)
        agent = build_agent(method, env, stage3, seed=42)
        agent.learn(total_timesteps=64)
        obs, _ = env.reset(seed=0)
        action, _ = agent.predict(obs, deterministic=True)
        a = float(np.asarray(action).reshape(-1)[0])
        assert (
            env_cfg.exposure_min - 1e-5 <= a <= env_cfg.exposure_max + 1e-5
        )
