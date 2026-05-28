"""Robust-InVAR-RL Phase 2: post-hoc score calibration.

Two calibrators map raw ranking scores (or any 1-D score-like
statistic) into a probability of profitable wrapper performance:

- :class:`PlattCalibrator`: 1-D logistic regression on
  ``(score -> profitable_indicator)``. Implemented via
  :class:`sklearn.linear_model.LogisticRegression`.
- :class:`IsotonicCalibrator`: non-parametric monotone fit via
  :class:`sklearn.isotonic.IsotonicRegression`.

Both expose a common interface::

    cal = PlattCalibrator().fit(scores, labels_binary)
    p = cal.predict_proba(new_scores)  # ndarray in [0, 1]

References:
- Platt 1999, "Probabilistic Outputs for SVMs and Comparisons to
  Regularized Likelihood Methods" (Adv. Large Margin Classifiers).
- Zadrozny + Elkan 2002, "Transforming Classifier Scores into
  Accurate Multiclass Probability Estimates" (KDD).

No silent fallbacks: invalid inputs raise ``ValueError`` with an
``[ERR]`` prefix. ``fit`` and ``predict_proba`` are deterministic
given the same inputs.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator(Protocol):
    """Common calibrator interface used by :mod:`prior_exposure`."""

    def fit(
        self, scores: np.ndarray, labels_binary: np.ndarray
    ) -> "Calibrator":
        ...

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        ...


def _validate_fit_inputs(
    scores: np.ndarray, labels_binary: np.ndarray
) -> None:
    if scores.ndim != 1:
        raise ValueError(
            f"[ERR] scores must be 1D; got {scores.ndim}D"
        )
    if labels_binary.ndim != 1:
        raise ValueError(
            f"[ERR] labels_binary must be 1D; got {labels_binary.ndim}D"
        )
    if scores.shape[0] != labels_binary.shape[0]:
        raise ValueError(
            "[ERR] scores and labels_binary length mismatch: "
            f"{scores.shape[0]} vs {labels_binary.shape[0]}"
        )
    if scores.shape[0] < 4:
        raise ValueError(
            f"[ERR] need >= 4 fit samples; got {scores.shape[0]}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("[ERR] scores contain NaN or inf")
    uniq = np.unique(labels_binary)
    if not set(uniq.tolist()).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"[ERR] labels_binary must be in {{0, 1}}; got uniques={uniq.tolist()}"
        )
    if uniq.size < 2:
        raise ValueError(
            "[ERR] labels_binary must contain both classes for fit; "
            f"got only {uniq.tolist()}"
        )


class PlattCalibrator:
    """1-D logistic calibrator (Platt scaling).

    Wraps :class:`sklearn.linear_model.LogisticRegression` with
    ``solver='lbfgs'`` and no regularisation rescale (C=1). Inputs are
    reshaped to ``(n, 1)`` internally. The output of
    :meth:`predict_proba` is the probability of class 1 (profitable).
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        if C <= 0.0:
            raise ValueError(f"[ERR] C must be > 0; got {C}")
        if max_iter < 1:
            raise ValueError(
                f"[ERR] max_iter must be >= 1; got {max_iter}"
            )
        self._model = LogisticRegression(
            C=float(C),
            max_iter=int(max_iter),
            solver="lbfgs",
        )
        self._fitted = False

    def fit(
        self, scores: np.ndarray, labels_binary: np.ndarray
    ) -> "PlattCalibrator":
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels_binary = np.asarray(labels_binary, dtype=np.int64).ravel()
        _validate_fit_inputs(scores, labels_binary)
        self._model.fit(scores.reshape(-1, 1), labels_binary)
        self._fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "[ERR] PlattCalibrator.predict_proba called before fit"
            )
        s = np.asarray(scores, dtype=np.float64).ravel()
        if s.size == 0:
            return np.zeros(0, dtype=np.float64)
        if not np.isfinite(s).all():
            raise ValueError("[ERR] scores contain NaN or inf")
        proba = self._model.predict_proba(s.reshape(-1, 1))[:, 1]
        return np.clip(proba.astype(np.float64), 0.0, 1.0)


class IsotonicCalibrator:
    """Non-parametric monotone calibrator (isotonic regression).

    Wraps :class:`sklearn.isotonic.IsotonicRegression` with
    ``out_of_bounds='clip'``; predictions outside the fitted score
    range are pinned to the boundary values.
    """

    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(
        self, scores: np.ndarray, labels_binary: np.ndarray
    ) -> "IsotonicCalibrator":
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels_binary = np.asarray(labels_binary, dtype=np.float64).ravel()
        _validate_fit_inputs(scores, labels_binary.astype(np.int64))
        self._model.fit(scores, labels_binary)
        self._fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "[ERR] IsotonicCalibrator.predict_proba called before fit"
            )
        s = np.asarray(scores, dtype=np.float64).ravel()
        if s.size == 0:
            return np.zeros(0, dtype=np.float64)
        if not np.isfinite(s).all():
            raise ValueError("[ERR] scores contain NaN or inf")
        proba = self._model.predict(s)
        return np.clip(proba.astype(np.float64), 0.0, 1.0)


def build_calibrator(method: str) -> Calibrator:
    """Factory for the two supported calibrators.

    Args:
        method: ``"platt"`` or ``"isotonic"``.

    Returns:
        An unfitted :class:`Calibrator` instance.
    """
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(
        f"[ERR] unknown calibration method {method!r}; "
        "expected 'platt' or 'isotonic'"
    )


__all__ = [
    "Calibrator",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "build_calibrator",
]
