"""Robust-InVAR-RL Phase 2: Kelly-style sizing prior over wrapper PnL.

Per the source design doc (2026-05-26), the residual-SAC controller
overlays a Kelly-style prior exposure ``e_star_t`` on top of the
wrapper. The prior is

    e_star_t = clip(kappa * mu_hat_t / (sigma_hat_t^2 + eps), 0, e_max)

with both ``mu_hat`` and ``sigma_hat`` derived from observable,
calibrated wrapper statistics (NOT from a separate forecasting model):

- ``mu_hat_t`` combines a top-bottom score spread, the calibrated
  probability of profitable wrapper performance, and a confidence
  scale derived from L1 score uncertainty. The combination is a
  weighted sum normalised to roughly ``[-1, +1]``.
- ``sigma_hat_t`` is the EWMA volatility of recent wrapper strategy
  PnL with a 21-day half-life (the standard finance EWMA convention).

The prior is interpretable and deliberately small: it gives the SAC
residual a sensible anchor where canonical SAC was already near
optimal (F4/F5 in Phase 1), while still letting SAC add value where
its anchor is poor (F2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.models.robust_invar_rl.calibration import Calibrator


@dataclass
class KellySizingPriorConfig:
    """Hyperparameters for :class:`KellySizingPrior`.

    Notes on units (load-bearing):
    - ``mu_hat`` is computed in TWO stages. The combination of spread,
      calibrated probability, and confidence yields a dimensionless
      signal in roughly ``[-1, +1]`` (see :meth:`compute_mu_hat`).
      Per-tape conversion to expected-daily-return units is then
      performed by multiplying by ``mu_scale`` inside
      :func:`build_e_star_tape_from_aux`. ``mu_scale`` defaults to the
      mean absolute wrapper return on the calibrator (validation)
      segment, i.e. the typical daily |strategy return| magnitude.
    - ``sigma_hat`` is the EWMA standard deviation of recent wrapper
      strategy returns, in fractional daily-return units (e.g. 0.01 for
      a 1 percent daily move). Thus ``mu_hat / (sigma_hat^2 + eps)``
      gives a Kelly-style fraction in the same units as a leverage.
    - ``kappa`` is the Kelly fraction applied to the raw ratio. With
      ``kappa=1.0`` (full Kelly), even moderate Sharpe yields the
      ``e_max`` cap on every day; we ship a 5%-Kelly default so the
      prior actually varies across the test window.
    """

    kappa: float = 0.05
    e_max: float = 1.5
    eps: float = 1.0e-6
    ewma_half_life_days: int = 21
    spread_weight: float = 0.5
    prob_weight: float = 0.5
    confidence_weight: float = 0.2
    mu_clip: float = 1.0
    mu_scale_floor: float = 1.0e-4


def _ewma_alpha_from_half_life(half_life_days: int) -> float:
    if half_life_days < 1:
        raise ValueError(
            f"[ERR] ewma_half_life_days must be >= 1; got {half_life_days}"
        )
    # alpha = 1 - 0.5 ** (1 / hl); larger alpha = faster decay.
    return float(1.0 - 0.5 ** (1.0 / float(half_life_days)))


def _ewma_std(returns: np.ndarray, alpha: float) -> float:
    """EWMA standard deviation of a 1-D return series.

    Uses recursive ``v_t = (1 - alpha) * v_{t-1} + alpha * r_t^2`` over
    de-meaned returns to estimate variance, then sqrt. Returns 0.0 when
    fewer than 2 observations are available.
    """
    if returns.size < 2:
        return 0.0
    mu = float(returns.mean())
    dev2 = (returns - mu) ** 2
    v = 0.0
    init = False
    for x in dev2:
        if not init:
            v = float(x)
            init = True
        else:
            v = (1.0 - alpha) * v + alpha * float(x)
    return float(np.sqrt(max(v, 0.0)))


def _topk_means(scores: np.ndarray, mask: np.ndarray, K: int) -> tuple:
    """Compute (top-K mean, bottom-K mean) over the active scores."""
    active = mask.astype(bool)
    s = scores[active]
    if s.size < 2:
        return 0.0, 0.0
    K_eff = int(max(1, min(K, s.size // 2)))
    order = np.argsort(s)
    bot = float(s[order[:K_eff]].mean())
    top = float(s[order[-K_eff:]].mean())
    return top, bot


class KellySizingPrior:
    """Per-day Kelly-style sizing prior.

    ``e_star_t = clip(kappa * mu_hat_t / (sigma_hat_t^2 + eps), 0, e_max)``

    where ``mu_hat_t`` and ``sigma_hat_t`` are computed from observable
    wrapper statistics (calibrated by the validation segment). The
    prior is intentionally bounded into ``[0, e_max]``: residual SAC
    is the only mechanism that can cap or extend exposure beyond this
    anchor.
    """

    def __init__(
        self,
        config: Optional[KellySizingPriorConfig] = None,
        kappa: Optional[float] = None,
        e_max: Optional[float] = None,
        eps: Optional[float] = None,
    ) -> None:
        if config is None:
            config = KellySizingPriorConfig()
        # CLI override hatch for the three load-bearing knobs.
        if kappa is not None:
            config = KellySizingPriorConfig(
                kappa=float(kappa), e_max=config.e_max, eps=config.eps,
                ewma_half_life_days=config.ewma_half_life_days,
                spread_weight=config.spread_weight,
                prob_weight=config.prob_weight,
                confidence_weight=config.confidence_weight,
                mu_clip=config.mu_clip,
                mu_scale_floor=config.mu_scale_floor,
            )
        if e_max is not None:
            config = KellySizingPriorConfig(
                kappa=config.kappa, e_max=float(e_max), eps=config.eps,
                ewma_half_life_days=config.ewma_half_life_days,
                spread_weight=config.spread_weight,
                prob_weight=config.prob_weight,
                confidence_weight=config.confidence_weight,
                mu_clip=config.mu_clip,
                mu_scale_floor=config.mu_scale_floor,
            )
        if eps is not None:
            config = KellySizingPriorConfig(
                kappa=config.kappa, e_max=config.e_max, eps=float(eps),
                ewma_half_life_days=config.ewma_half_life_days,
                spread_weight=config.spread_weight,
                prob_weight=config.prob_weight,
                confidence_weight=config.confidence_weight,
                mu_clip=config.mu_clip,
                mu_scale_floor=config.mu_scale_floor,
            )
        if config.kappa <= 0.0:
            raise ValueError(f"[ERR] kappa must be > 0; got {config.kappa}")
        if config.e_max <= 0.0:
            raise ValueError(f"[ERR] e_max must be > 0; got {config.e_max}")
        if config.eps <= 0.0:
            raise ValueError(f"[ERR] eps must be > 0; got {config.eps}")
        self._cfg = config
        self._ewma_alpha = _ewma_alpha_from_half_life(
            config.ewma_half_life_days
        )

    @property
    def cfg(self) -> KellySizingPriorConfig:
        return self._cfg

    def compute_mu_hat(
        self,
        scores_t: np.ndarray,
        mask_t: np.ndarray,
        K: int,
        calibrator: Calibrator,
        l1_uncertainty_t: float,
    ) -> float:
        """Calibrated, normalised expected wrapper edge for day ``t``.

        ``mu_hat`` = ``spread_weight * normalised_top_bottom_spread
        + prob_weight * (2 * p_calibrated - 1) +
        confidence_weight * confidence_scale``

        - normalised_top_bottom_spread: ``(top_K_mean - bottom_K_mean)``
          divided by the cross-sectional score scale, clipped to
          ``[-1, +1]``. Captures the directional ranker signal.
        - p_calibrated: probability of profitable wrapper performance
          from the validation-fitted calibrator, applied to the
          top-bottom-spread itself. Mapped from ``[0, 1]`` to
          ``[-1, +1]`` so its zero point is "no edge".
        - confidence_scale: ``1 - tanh(l1_uncertainty)`` so larger L1
          uncertainty shrinks ``mu_hat`` towards zero.

        The output is clipped to ``[-mu_clip, +mu_clip]`` to keep the
        Kelly numerator bounded.
        """
        scores_t = np.asarray(scores_t, dtype=np.float64).ravel()
        mask_t = np.asarray(mask_t).ravel()
        if scores_t.shape != mask_t.shape:
            raise ValueError(
                "[ERR] scores_t and mask_t shape mismatch: "
                f"{scores_t.shape} vs {mask_t.shape}"
            )
        if scores_t.size == 0:
            return 0.0
        top_mean, bot_mean = _topk_means(scores_t, mask_t, K)
        spread = top_mean - bot_mean
        active = mask_t.astype(bool)
        active_scores = scores_t[active]
        if active_scores.size >= 2:
            scale = float(active_scores.std(ddof=1))
        else:
            scale = 0.0
        if scale <= 1.0e-12:
            spread_norm = 0.0
        else:
            spread_norm = float(np.clip(spread / scale, -1.0, 1.0))
        # Calibrated probability evaluated AT THE SPREAD value: the
        # validation calibrator was fit on (spread -> profitable),
        # so we pass the day's spread through it.
        p = float(
            calibrator.predict_proba(np.asarray([spread], dtype=np.float64))[0]
        )
        prob_signed = 2.0 * p - 1.0
        unc = float(l1_uncertainty_t)
        conf_scale = float(1.0 - np.tanh(max(0.0, unc)))
        mu_hat = (
            self._cfg.spread_weight * spread_norm
            + self._cfg.prob_weight * prob_signed
            + self._cfg.confidence_weight * conf_scale * np.sign(prob_signed)
        )
        return float(np.clip(mu_hat, -self._cfg.mu_clip, self._cfg.mu_clip))

    def compute_sigma_hat(
        self, recent_strategy_returns: np.ndarray
    ) -> float:
        """EWMA volatility of recent wrapper strategy returns."""
        arr = np.asarray(
            recent_strategy_returns, dtype=np.float64
        ).ravel()
        return _ewma_std(arr, self._ewma_alpha)

    def compute_e_star(self, mu_hat: float, sigma_hat: float) -> float:
        """Kelly-style exposure prior, clipped to ``[0, e_max]``.

        ``e_star = clip(kappa * mu_hat / (sigma_hat^2 + eps), 0, e_max)``

        When ``mu_hat <= 0`` the prior pins exposure to 0 (we do not
        recommend going net-short via the prior; SAC's residual is what
        can push exposure above 0 when the prior is conservative).

        If ``sigma_hat`` is below ``mu_scale_floor`` (a cold-start guard
        for the first day or any all-zero return window), the prior is
        pinned to 0 rather than allowing ``eps`` to drive the
        denominator and saturate the clip.
        """
        if not np.isfinite(mu_hat):
            raise ValueError(
                f"[ERR] mu_hat must be finite; got {mu_hat}"
            )
        if not np.isfinite(sigma_hat) or sigma_hat < 0.0:
            raise ValueError(
                f"[ERR] sigma_hat must be finite and >= 0; got {sigma_hat}"
            )
        if sigma_hat < self._cfg.mu_scale_floor:
            return 0.0
        denom = sigma_hat * sigma_hat + self._cfg.eps
        raw = self._cfg.kappa * mu_hat / denom
        return float(np.clip(raw, 0.0, self._cfg.e_max))


def build_e_star_tape(
    prior: KellySizingPrior,
    scores_per_day: np.ndarray,
    mask_per_day: np.ndarray,
    wrapper_returns: np.ndarray,
    K: int,
    calibrator: Calibrator,
    l1_uncertainty_per_day: np.ndarray,
    vol_window_days: int = 21,
) -> np.ndarray:
    """Build the per-day ``e_star`` tape for a full episode window.

    Args:
        prior: Configured Kelly sizing prior.
        scores_per_day: ``(T, N)`` per-day per-stock L1 scores.
        mask_per_day: ``(T, N)`` per-day per-stock active mask.
        wrapper_returns: ``(T,)`` realised wrapper PnL series; used as
            the running sigma input.
        K: Wrapper per-side K.
        calibrator: Validation-fitted calibrator.
        l1_uncertainty_per_day: ``(T,)`` L1 score uncertainty (per-day
            ``std`` of active scores, or any monotone proxy).
        vol_window_days: Rolling window length supplied to
            ``compute_sigma_hat`` (defaults to the canonical 21 trading
            days; matches the EWMA half-life convention).

    Returns:
        ``(T,)`` ndarray of ``e_star`` values, each in ``[0, e_max]``.
    """
    T = int(scores_per_day.shape[0])
    if mask_per_day.shape[0] != T:
        raise ValueError(
            "[ERR] mask_per_day length mismatch: "
            f"{mask_per_day.shape[0]} vs {T}"
        )
    if wrapper_returns.shape[0] != T:
        raise ValueError(
            "[ERR] wrapper_returns length mismatch: "
            f"{wrapper_returns.shape[0]} vs {T}"
        )
    if l1_uncertainty_per_day.shape[0] != T:
        raise ValueError(
            "[ERR] l1_uncertainty_per_day length mismatch: "
            f"{l1_uncertainty_per_day.shape[0]} vs {T}"
        )
    if vol_window_days < 2:
        raise ValueError(
            f"[ERR] vol_window_days must be >= 2; got {vol_window_days}"
        )
    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        lo = max(0, t - vol_window_days + 1)
        window = wrapper_returns[lo:t + 1] if t > 0 else np.asarray([0.0])
        sigma_hat = prior.compute_sigma_hat(window)
        mu_hat = prior.compute_mu_hat(
            scores_t=scores_per_day[t],
            mask_t=mask_per_day[t],
            K=K,
            calibrator=calibrator,
            l1_uncertainty_t=float(l1_uncertainty_per_day[t]),
        )
        out[t] = prior.compute_e_star(mu_hat, sigma_hat)
    return out


def build_e_star_tape_from_aux(
    prior: KellySizingPrior,
    score_spread_topk: np.ndarray,
    score_uncertainty: np.ndarray,
    wrapper_returns: np.ndarray,
    calibrator: Calibrator,
    vol_window_days: int = 21,
    mu_scale: Optional[float] = None,
) -> np.ndarray:
    """Compact ``e_star`` tape builder for the Phase 2 aux representation.

    Unlike :func:`build_e_star_tape`, this variant consumes already
    aggregated per-day statistics (top-K spread + score-std uncertainty)
    rather than the raw ``(T, N)`` score panel. It is the recommended
    entry point when the per-day spread has been precomputed via
    :func:`invar_rl.layer3_control.phase2_precompute.compute_phase2_aux`.

    The ``mu_hat`` for day ``t`` is::

        spread_norm   = clip(spread_t / spread_scale, -1, +1)
        prob_signed   = 2 * calibrator(spread_t) - 1
        confidence    = 1 - tanh(uncertainty_t)
        mu_signal     = spread_weight * spread_norm
                      + prob_weight * prob_signed
                      + confidence_weight * confidence * sign(prob_signed)
        mu_hat        = mu_signal * mu_scale          # daily return units

    where ``spread_scale`` is the in-sample standard deviation of
    ``score_spread_topk`` (a stable per-tape normaliser) and
    ``mu_scale`` converts the dimensionless ``mu_signal`` in
    ``[-1, +1]`` into expected-daily-return units (so the Kelly ratio
    ``mu_hat / sigma_hat^2`` is dimensionally consistent). When
    ``mu_scale`` is left at the default of ``None``, it is set to the
    mean absolute ``wrapper_returns`` over the input window (a stable
    per-tape estimate of typical daily |strategy return|), floored at
    ``prior.cfg.mu_scale_floor`` to avoid degenerate zeros.

    Args:
        prior: Configured :class:`KellySizingPrior` instance.
        score_spread_topk: ``(T,)`` per-day top-K minus bottom-K means.
        score_uncertainty: ``(T,)`` per-day score-std proxy for L1
            uncertainty.
        wrapper_returns: ``(T,)`` realised wrapper PnL series.
        calibrator: Validation-fitted calibrator.
        vol_window_days: Rolling-window length for ``sigma_hat``.
        mu_scale: Per-tape conversion factor from dimensionless
            ``mu_signal`` to expected daily fractional return. When
            ``None`` (default), is derived from ``wrapper_returns`` as
            ``max(mean(|wrapper_returns|), cfg.mu_scale_floor)``. Callers
            that want a fixed cross-fold scale (e.g. fit on the
            calibrator/val segment) should pass it explicitly.

    Returns:
        ``(T,)`` ndarray of ``e_star`` values in ``[0, e_max]``.
    """
    spread = np.asarray(score_spread_topk, dtype=np.float64).ravel()
    unc = np.asarray(score_uncertainty, dtype=np.float64).ravel()
    rets = np.asarray(wrapper_returns, dtype=np.float64).ravel()
    T = spread.shape[0]
    if unc.shape[0] != T or rets.shape[0] != T:
        raise ValueError(
            "[ERR] aux arrays length mismatch: "
            f"spread={T} unc={unc.shape[0]} rets={rets.shape[0]}"
        )
    if vol_window_days < 2:
        raise ValueError(
            f"[ERR] vol_window_days must be >= 2; got {vol_window_days}"
        )
    if T == 0:
        return np.zeros(0, dtype=np.float64)

    spread_scale = float(spread.std(ddof=1)) if T >= 2 else 0.0
    cfg = prior.cfg
    if mu_scale is None:
        mean_abs = float(np.mean(np.abs(rets))) if rets.size > 0 else 0.0
        mu_scale_eff = float(max(mean_abs, cfg.mu_scale_floor))
    else:
        if not np.isfinite(float(mu_scale)) or float(mu_scale) <= 0.0:
            raise ValueError(
                f"[ERR] mu_scale must be > 0 and finite; got {mu_scale}"
            )
        mu_scale_eff = float(max(float(mu_scale), cfg.mu_scale_floor))
    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        if spread_scale > 1.0e-12:
            spread_norm = float(np.clip(spread[t] / spread_scale, -1.0, 1.0))
        else:
            spread_norm = 0.0
        p = float(
            calibrator.predict_proba(
                np.asarray([spread[t]], dtype=np.float64)
            )[0]
        )
        prob_signed = 2.0 * p - 1.0
        conf_scale = float(1.0 - np.tanh(max(0.0, float(unc[t]))))
        mu_signal = (
            cfg.spread_weight * spread_norm
            + cfg.prob_weight * prob_signed
            + cfg.confidence_weight * conf_scale * np.sign(prob_signed)
        )
        mu_signal = float(np.clip(mu_signal, -cfg.mu_clip, cfg.mu_clip))
        # Convert dimensionless [-1, +1] signal to daily-return units.
        mu_hat = float(mu_signal * mu_scale_eff)
        lo = max(0, t - vol_window_days + 1)
        if t == 0:
            sigma_hat = 0.0
        else:
            sigma_hat = prior.compute_sigma_hat(rets[lo:t + 1])
        out[t] = prior.compute_e_star(mu_hat, sigma_hat)
    return out


__all__ = [
    "KellySizingPriorConfig",
    "KellySizingPrior",
    "build_e_star_tape",
    "build_e_star_tape_from_aux",
]
