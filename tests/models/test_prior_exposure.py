"""Tests for Phase 2 Kelly sizing prior."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.robust_invar_rl.calibration import PlattCalibrator
from src.models.robust_invar_rl.prior_exposure import (
    KellySizingPrior,
    KellySizingPriorConfig,
    build_e_star_tape_from_aux,
)


def _fit_dummy_calibrator() -> PlattCalibrator:
    rng = np.random.default_rng(0)
    n = 200
    pos = rng.normal(0.5, 0.2, size=n // 2)
    neg = rng.normal(-0.5, 0.2, size=n // 2)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate(
        [np.ones(n // 2, dtype=np.int64), np.zeros(n // 2, dtype=np.int64)]
    )
    return PlattCalibrator().fit(scores, labels)


def test_e_star_within_bounds() -> None:
    prior = KellySizingPrior(KellySizingPriorConfig(kappa=1.0, e_max=1.5))
    e = prior.compute_e_star(mu_hat=0.5, sigma_hat=0.05)
    assert 0.0 <= e <= 1.5


def test_e_star_pins_to_zero_on_negative_mu() -> None:
    prior = KellySizingPrior()
    e = prior.compute_e_star(mu_hat=-0.7, sigma_hat=0.05)
    assert e == 0.0


def test_e_star_clips_at_e_max_on_strong_signal() -> None:
    prior = KellySizingPrior(KellySizingPriorConfig(kappa=10.0, e_max=1.5))
    e = prior.compute_e_star(mu_hat=1.0, sigma_hat=0.001)
    assert e == pytest.approx(1.5)


def test_sigma_hat_is_positive_on_volatile_series() -> None:
    prior = KellySizingPrior()
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0, 0.01, size=30)
    s = prior.compute_sigma_hat(rets)
    assert s > 0.0


def test_mu_hat_finite_on_calibrated_inputs() -> None:
    prior = KellySizingPrior()
    cal = _fit_dummy_calibrator()
    n = 30
    scores = np.random.default_rng(2).normal(0.0, 1.0, size=n)
    mask = np.ones(n, dtype=bool)
    mu = prior.compute_mu_hat(
        scores_t=scores, mask_t=mask, K=5,
        calibrator=cal, l1_uncertainty_t=0.5,
    )
    assert np.isfinite(mu)
    assert -1.0 <= mu <= 1.0


def test_build_e_star_tape_from_aux_shape_and_bounds() -> None:
    prior = KellySizingPrior(KellySizingPriorConfig(kappa=1.0, e_max=1.5))
    cal = _fit_dummy_calibrator()
    T = 60
    rng = np.random.default_rng(3)
    spread = rng.normal(0.2, 0.5, size=T)
    unc = np.abs(rng.normal(0.5, 0.1, size=T))
    rets = rng.normal(0.0, 0.005, size=T)
    e_star = build_e_star_tape_from_aux(
        prior=prior,
        score_spread_topk=spread,
        score_uncertainty=unc,
        wrapper_returns=rets,
        calibrator=cal,
    )
    assert e_star.shape == (T,)
    assert (e_star >= 0.0).all()
    assert (e_star <= 1.5 + 1.0e-9).all()
    assert np.isfinite(e_star).all()


def test_kappa_zero_rejected() -> None:
    with pytest.raises(ValueError):
        KellySizingPrior(KellySizingPriorConfig(kappa=0.0, e_max=1.5))


def test_negative_sigma_rejected() -> None:
    prior = KellySizingPrior()
    with pytest.raises(ValueError):
        prior.compute_e_star(mu_hat=0.3, sigma_hat=-0.1)
