"""Tests for Layer 3, the exposure-control environment (no agent yet)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from invar_rl.common.config import load_layer3_config
from invar_rl.common.seeding import make_rng
from invar_rl.layer3_control.env import ExposureEnv
from invar_rl.layer3_control.observation import RiskState, build_observation
from invar_rl.layer3_control.precompute import EpisodeTape
from invar_rl.layer3_control.reward import DifferentialSharpe

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _fake_tape(t: int = 300, d: int = 6, seed: int = 42) -> EpisodeTape:
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


def _cfg():
    return load_layer3_config(CONFIG_DIR / "layer3.yaml")


def test_conforms_to_gymnasium_api() -> None:
    env = ExposureEnv(_fake_tape(), _cfg())
    check_env(env, skip_render_check=True)


def test_constant_exposure_reproduces_buy_and_hold() -> None:
    tape = _fake_tape()
    cfg = _cfg()
    env = ExposureEnv(tape, cfg)
    env.reset(seed=0)
    c = 0.2  # reachable from exposure_min=0 within the change band in step 1
    equity = 1.0
    steps = 0
    while True:
        _, _, term, trunc, info = env.step(np.array([c], dtype=np.float32))
        equity *= 1.0 + c * float(tape.base_return[steps])
        assert abs(info["equity"] - equity) < 1e-6
        steps += 1
        if term or trunc:
            break
    assert steps > 10


def test_random_policy_runs_and_is_reproducible() -> None:
    cfg = _cfg()

    def rollout() -> float:
        env = ExposureEnv(_fake_tape(), cfg)
        env.reset(seed=123)
        env.action_space.seed(123)
        total = 0.0
        while True:
            a = env.action_space.sample()
            _, r, term, trunc, _ = env.step(a)
            assert np.isfinite(r)
            total += r
            if term or trunc:
                break
        return total

    assert abs(rollout() - rollout()) < 1e-9


def test_differential_sharpe_tracks_direct_sharpe() -> None:
    rng = make_rng(7)
    mu, sigma = 0.001, 0.01
    stream = rng.normal(mu, sigma, size=20000)
    ds = DifferentialSharpe(eta=0.001)
    for r in stream:
        ds.update(float(r))
    ema_sharpe = ds._a / np.sqrt(ds._b - ds._a ** 2)
    direct = stream.mean() / stream.std()
    assert abs(ema_sharpe - direct) < 0.15 * abs(direct) + 0.02


def test_differential_sharpe_is_bounded_on_low_variance_stream() -> None:
    # A near-constant return stream drives B - A**2 toward zero; without
    # the variance floor and increment clip this produces the exploding,
    # penalty-incomparable rewards seen in the first Stage 3 sweep.
    rng = make_rng(11)
    stream = 0.01 + rng.normal(0.0, 1e-7, size=5000)
    ds = DifferentialSharpe(eta=0.01, variance_floor=1e-6, clip=5.0)
    increments = [ds.update(float(r)) for r in stream]
    arr = np.asarray(increments)
    assert np.isfinite(arr).all()
    assert np.abs(arr).max() <= 5.0 + 1e-9


def test_observation_has_no_future_information() -> None:
    tape = _fake_tape()
    t = 100
    risk = RiskState(rolling_vol=0.01, drawdown=0.05, exposure=0.3)
    base_obs = build_observation(tape, t, risk)

    future = copy.deepcopy(tape)
    future.score_dispersion[t + 1:] = 999.0
    future.macro_encoding[t + 1:] = -999.0
    future.pred_vol[t + 1:] = 999.0
    future.base_return[t + 1:] = 999.0
    assert np.array_equal(base_obs, build_observation(future, t, risk))
