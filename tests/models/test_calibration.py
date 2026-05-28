"""Tests for Phase 2 calibration module."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.robust_invar_rl.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    build_calibrator,
)


def _make_well_separated(n: int = 200, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    pos = rng.normal(1.0, 0.5, size=n // 2)
    neg = rng.normal(-1.0, 0.5, size=n // 2)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate(
        [np.ones(n // 2, dtype=np.int64), np.zeros(n // 2, dtype=np.int64)]
    )
    perm = rng.permutation(n)
    return scores[perm], labels[perm]


def test_platt_returns_high_proba_for_positive_score() -> None:
    scores, labels = _make_well_separated()
    cal = PlattCalibrator().fit(scores, labels)
    p_high = cal.predict_proba(np.asarray([2.0]))[0]
    p_low = cal.predict_proba(np.asarray([-2.0]))[0]
    assert p_high > 0.8
    assert p_low < 0.2


def test_isotonic_monotone_on_unseen_scores() -> None:
    scores, labels = _make_well_separated()
    cal = IsotonicCalibrator().fit(scores, labels)
    probs = cal.predict_proba(np.linspace(-3.0, 3.0, 25))
    # Isotonic fit + clip => monotonically non-decreasing.
    diffs = np.diff(probs)
    assert (diffs >= -1.0e-9).all(), f"non-monotone diffs={diffs}"


def test_platt_proba_in_unit_interval() -> None:
    scores, labels = _make_well_separated()
    cal = PlattCalibrator().fit(scores, labels)
    probs = cal.predict_proba(np.linspace(-10.0, 10.0, 50))
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_build_calibrator_factory() -> None:
    cal_p = build_calibrator("platt")
    cal_i = build_calibrator("isotonic")
    assert isinstance(cal_p, PlattCalibrator)
    assert isinstance(cal_i, IsotonicCalibrator)
    with pytest.raises(ValueError):
        build_calibrator("nonsense")


def test_fit_rejects_singleton_class() -> None:
    cal = PlattCalibrator()
    with pytest.raises(ValueError):
        cal.fit(
            np.linspace(-1.0, 1.0, 10),
            np.zeros(10, dtype=np.int64),
        )


def test_fit_rejects_nan_scores() -> None:
    scores, labels = _make_well_separated(n=20)
    scores[0] = np.nan
    cal = PlattCalibrator()
    with pytest.raises(ValueError):
        cal.fit(scores, labels)


def test_predict_proba_before_fit_raises() -> None:
    cal = PlattCalibrator()
    with pytest.raises(RuntimeError):
        cal.predict_proba(np.asarray([0.0]))
