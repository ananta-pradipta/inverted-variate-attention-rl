"""Tests for Phase 3 CompactObservationBuilder."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.robust_invar_rl.compact_obs import (
    DRAWDOWN_WINDOW_DAYS,
    HIT_RATE_WINDOW_DAYS,
    N_BASE_FIELDS,
    N_REGIME_CLUSTERS,
    TURNOVER_WINDOW_DAYS,
    CompactObservationBuilder,
    CompactObservationConfig,
    CompactObservationTape,
    build_regime_one_hot,
)


def _make_tape(
    T: int = 60, with_regime: bool = True, seed: int = 0
) -> CompactObservationTape:
    rng = np.random.default_rng(seed)
    p_hat = rng.uniform(0.3, 0.7, size=T).astype(np.float64)
    mu_hat = rng.normal(0.0, 0.005, size=T).astype(np.float64)
    sigma_hat = rng.uniform(0.005, 0.02, size=T).astype(np.float64)
    e_star = rng.uniform(0.3, 1.5, size=T).astype(np.float64)
    vix = rng.uniform(10.0, 35.0, size=T).astype(np.float64)
    ust10y = rng.uniform(1.0, 4.5, size=T).astype(np.float64)
    regime = None
    if with_regime:
        ids = rng.integers(0, N_REGIME_CLUSTERS, size=T)
        regime = np.zeros((T, N_REGIME_CLUSTERS), dtype=np.float32)
        regime[np.arange(T), ids] = 1.0
    return CompactObservationTape(
        p_hat=p_hat,
        mu_hat=mu_hat,
        sigma_hat=sigma_hat,
        e_star=e_star,
        vix_per_day=vix,
        ust10y_per_day=ust10y,
        regime_one_hot=regime,
    )


def test_obs_dim_with_regime_is_17() -> None:
    """8 base + 8 regime + 1 e_star = 17 (spec count, all toggles on)."""
    tape = _make_tape(T=30, with_regime=True)
    cfg = CompactObservationConfig(include_regime_one_hot=True)
    builder = CompactObservationBuilder(tape=tape, cfg=cfg)
    assert builder.obs_dim == 17
    assert builder.obs_dim == N_BASE_FIELDS + N_REGIME_CLUSTERS + 1


def test_obs_dim_without_regime_is_9() -> None:
    """8 base + 0 regime + 1 e_star = 9 (regime toggle off)."""
    tape = _make_tape(T=30, with_regime=False)
    cfg = CompactObservationConfig(include_regime_one_hot=True)
    builder = CompactObservationBuilder(tape=tape, cfg=cfg)
    # Regime is None in tape; builder should silently drop it.
    assert builder.obs_dim == 9
    assert builder.has_regime is False


def test_obs_dim_with_toggle_off_drops_regime() -> None:
    tape = _make_tape(T=30, with_regime=True)
    cfg = CompactObservationConfig(include_regime_one_hot=False)
    builder = CompactObservationBuilder(tape=tape, cfg=cfg)
    assert builder.obs_dim == 9


def test_build_returns_float32_finite_correct_shape() -> None:
    tape = _make_tape(T=30, with_regime=True)
    builder = CompactObservationBuilder(tape=tape)
    builder.reset()
    obs = builder.build(step_idx=0)
    assert obs.dtype == np.float32
    assert obs.shape == (17,)
    assert np.isfinite(obs).all()


def test_e_star_is_last_dim() -> None:
    """e_star MUST be appended last to mirror KellyPriorEnvWrapper."""
    tape = _make_tape(T=10, with_regime=True)
    builder = CompactObservationBuilder(tape=tape)
    builder.reset()
    obs = builder.build(step_idx=3)
    assert float(obs[-1]) == pytest.approx(float(tape.e_star[3]))


def test_vix_and_ust10y_normalisation() -> None:
    tape = _make_tape(T=5, with_regime=False)
    cfg = CompactObservationConfig(
        vix_scale=20.0, ust10y_scale=4.0,
        include_regime_one_hot=False,
    )
    builder = CompactObservationBuilder(tape=tape, cfg=cfg)
    builder.reset()
    obs = builder.build(step_idx=2)
    # Index 6 is vix_normalised, index 7 is ust10y_normalised.
    assert float(obs[6]) == pytest.approx(
        float(tape.vix_per_day[2]) / 20.0, rel=1e-5,
    )
    assert float(obs[7]) == pytest.approx(
        float(tape.ust10y_per_day[2]) / 4.0, rel=1e-5,
    )


def test_drawdown_increases_after_losses() -> None:
    tape = _make_tape(T=30, with_regime=False)
    builder = CompactObservationBuilder(tape=tape)
    builder.reset()
    # Feed three losing days.
    for i, r in enumerate([-0.01, -0.02, -0.015]):
        builder.update(step_idx=i, strategy_return=r, exposure=1.0)
    obs = builder.build(step_idx=3)
    drawdown = float(obs[3])
    assert drawdown > 0.0


def test_hit_rate_starts_zero_then_tracks_wins() -> None:
    tape = _make_tape(T=10, with_regime=False)
    builder = CompactObservationBuilder(tape=tape)
    builder.reset()
    obs0 = builder.build(step_idx=0)
    assert float(obs0[5]) == 0.0
    builder.update(step_idx=0, strategy_return=0.01, exposure=1.0)
    builder.update(step_idx=1, strategy_return=0.02, exposure=1.0)
    obs2 = builder.build(step_idx=2)
    assert float(obs2[5]) == pytest.approx(1.0)


def test_turnover_tracks_exposure_changes() -> None:
    tape = _make_tape(T=10, with_regime=False)
    builder = CompactObservationBuilder(tape=tape)
    builder.reset()
    builder.update(step_idx=0, strategy_return=0.0, exposure=0.5)
    builder.update(step_idx=1, strategy_return=0.0, exposure=1.0)
    obs2 = builder.build(step_idx=2)
    # turnover_hist = [0.5, 0.5]; mean = 0.5.
    assert float(obs2[4]) == pytest.approx(0.5)


def test_update_raises_on_nan_return() -> None:
    tape = _make_tape(T=5, with_regime=False)
    builder = CompactObservationBuilder(tape=tape)
    builder.reset()
    with pytest.raises(ValueError, match="non-finite"):
        builder.update(step_idx=0, strategy_return=np.nan, exposure=1.0)


def test_update_raises_on_out_of_range_step() -> None:
    tape = _make_tape(T=3, with_regime=False)
    builder = CompactObservationBuilder(tape=tape)
    builder.reset()
    with pytest.raises(ValueError, match="out of range"):
        builder.update(step_idx=5, strategy_return=0.0, exposure=1.0)


def test_build_clamps_step_idx_past_tape_end() -> None:
    tape = _make_tape(T=4, with_regime=False)
    builder = CompactObservationBuilder(tape=tape)
    builder.reset()
    obs_last = builder.build(step_idx=3)
    obs_past = builder.build(step_idx=99)
    np.testing.assert_array_equal(obs_last, obs_past)


def test_tape_validates_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        CompactObservationTape(
            p_hat=np.zeros(5),
            mu_hat=np.zeros(4),
            sigma_hat=np.zeros(5),
            e_star=np.zeros(5),
            vix_per_day=np.zeros(5),
            ust10y_per_day=np.zeros(5),
        )


def test_tape_validates_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        CompactObservationTape(
            p_hat=np.full(3, np.nan),
            mu_hat=np.zeros(3),
            sigma_hat=np.zeros(3),
            e_star=np.zeros(3),
            vix_per_day=np.zeros(3),
            ust10y_per_day=np.zeros(3),
        )


def test_tape_validates_regime_width() -> None:
    with pytest.raises(ValueError, match="N_REGIME_CLUSTERS|columns"):
        CompactObservationTape(
            p_hat=np.zeros(3),
            mu_hat=np.zeros(3),
            sigma_hat=np.zeros(3),
            e_star=np.zeros(3),
            vix_per_day=np.zeros(3),
            ust10y_per_day=np.zeros(3),
            regime_one_hot=np.zeros((3, 4), dtype=np.float32),
        )


def test_build_regime_one_hot_assigns_correct_columns() -> None:
    days = np.asarray([100, 101, 102], dtype=np.int64)
    mapping = {100: 0, 101: 7, 102: 3}
    oh = build_regime_one_hot(days, mapping, n_clusters=8)
    assert oh.shape == (3, 8)
    assert oh[0, 0] == 1.0
    assert oh[1, 7] == 1.0
    assert oh[2, 3] == 1.0
    assert oh.sum(axis=1).tolist() == [1.0, 1.0, 1.0]


def test_build_regime_one_hot_missing_day_is_zero_row() -> None:
    days = np.asarray([100, 200], dtype=np.int64)
    mapping = {100: 5}
    oh = build_regime_one_hot(days, mapping, n_clusters=8)
    assert oh[0, 5] == 1.0
    assert oh[1].sum() == 0.0


def test_build_regime_one_hot_raises_on_out_of_range_cluster() -> None:
    days = np.asarray([100], dtype=np.int64)
    mapping = {100: 99}
    with pytest.raises(ValueError, match="out of range"):
        build_regime_one_hot(days, mapping, n_clusters=8)


def test_config_rejects_bad_scales() -> None:
    with pytest.raises(ValueError):
        CompactObservationConfig(vix_scale=0.0)
    with pytest.raises(ValueError):
        CompactObservationConfig(ust10y_scale=-1.0)
    with pytest.raises(ValueError):
        CompactObservationConfig(drawdown_window=1)


def test_default_window_constants_match_spec() -> None:
    assert DRAWDOWN_WINDOW_DAYS == 21
    assert TURNOVER_WINDOW_DAYS == 5
    assert HIT_RATE_WINDOW_DAYS == 21
