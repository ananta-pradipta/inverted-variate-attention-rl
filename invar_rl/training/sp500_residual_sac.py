"""Robust-InVAR-RL Phase 2 driver: residual SAC + Kelly prior on SP500.

For a single (fold, seed) cell, this driver:

1. Loads a frozen Phase 1 group-DRO Layer 1 ckpt (the ranker), with
   an explicit ``--layer1-ckpt`` (or root path) so the L1 stays the
   anchor identified in the Phase 1 stop-gate review.
2. Builds the canonical equal-weight top-K wrapper tape (no QP).
3. Computes per-day score-spread + score-std + profitable indicator
   over the train, val, and test segments via
   :func:`invar_rl.layer3_control.phase2_precompute.compute_phase2_aux`.
4. Fits a calibrator (Platt or isotonic) on the validation segment
   only: ``(score_spread_topk -> base_return > 0)``.
5. Builds the per-day Kelly-style ``e_star`` tape on train, val, test
   via :func:`src.models.robust_invar_rl.prior_exposure.build_e_star_tape_from_aux`.
6. Wraps :class:`invar_rl.layer3_control.env.ExposureEnv` with
   :class:`invar_rl.layer3_control.kelly_prior_env.KellyPriorEnvWrapper`.
7. Trains SB3 SAC on the wrapped train env.
8. Evaluates the policy deterministically on the wrapped test env
   and saves a daily tape (date, exposure, residual_action, e_star,
   strategy_return, base_return) for the paired-bootstrap pipeline.

Outputs (per cell):
- ``<output_dir>/foldF_seedS.json`` summary JSON
- ``<daily_tape_dir>/sac/foldF_seedS.parquet`` daily series
- ``<output_dir>/_ckpt/sac_f{F}_s{S}.zip`` saved SAC agent

Modes (``--mode``):
- ``e_star_only``: skip SAC training, evaluate using the prior alone
  (residual action = 0). Used by the smoke test to verify the prior
  by itself is sane.
- ``e_star_plus_sac`` (default): train residual SAC and evaluate.

CLI flags (canonical):
- ``--fold``, ``--seed``, ``--layer1-ckpt`` (or ``--layer1-ckpt-root``)
- ``--output-dir-root``, ``--daily-tape-dir``
- ``--delta-cap`` (default 0.25)
- ``--kappa`` (default 1.0)
- ``--e-max`` (default 1.5)
- ``--calibration-method`` (default ``"platt"``)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.invar import InVARConfig

from invar_rl.common.config import (
    load_layer2_config,
    load_layer3_config,
    load_stage3_config,
)
from invar_rl.common.seeding import set_global_seed
from invar_rl.data.lattice_bridge import build_lattice_bridge
from invar_rl.layer1_ranker.canonical_runner import load_trained_invar
from invar_rl.layer3_control.env import ExposureEnv
from invar_rl.layer3_control.kelly_prior_env import KellyPriorEnvWrapper
from invar_rl.layer3_control.phase2_precompute import compute_phase2_aux
from invar_rl.layer3_control.phase3_env_wrappers import (
    CompactObservationWrapper,
    OnlineSharpeRewardWrapper,
)
from invar_rl.layer3_control.precompute_canonical import (
    precompute_tape_canonical,
)
from src.models.robust_invar_rl.calibration import build_calibrator
from src.models.robust_invar_rl.compact_obs import (
    CompactObservationConfig,
    CompactObservationTape,
    N_BASE_FIELDS,
    N_REGIME_CLUSTERS,
    build_regime_one_hot,
)
from src.models.robust_invar_rl.online_sharpe_reward import (
    OnlineSharpeRewardConfig,
)
from src.models.robust_invar_rl.prior_exposure import (
    KellySizingPrior,
    KellySizingPriorConfig,
    build_e_star_tape_from_aux,
)


_LOG_PREFIX = "[Phase2-SP500-ResidualSAC]"

# Macro parquet columns used by the Phase 3 compact obs.
_PHASE3_VIX_COL = "vix"
_PHASE3_UST10Y_COL = "dgs10"


def _load_raw_macro_per_day(
    cfg: InVARConfig,
    bridge,
    cols: Tuple[str, ...] = (_PHASE3_VIX_COL, _PHASE3_UST10Y_COL),
) -> Dict[str, np.ndarray]:
    """Read raw (unscaled) macro columns aligned to ``bridge.dates``.

    Returns a dict ``{col: (T,) ndarray}`` where T = len(bridge.dates).
    Missing values are forward-filled (limit 5) then 0-filled to keep
    the array finite. Raises if any requested column is absent from the
    parquet.
    """
    if cfg.panel_kind == "lattice_native":
        macro_path = Path(cfg.universal_macro_duration_parquet)
    elif cfg.panel_kind == "nasdaq100":
        macro_path = Path(cfg.nasdaq100_macro_duration_parquet)
    elif cfg.panel_kind == "djia30":
        macro_path = Path(cfg.djia30_macro_duration_parquet)
    elif cfg.panel_kind in ("biotech_nbi", "biotech_nbi_enriched"):
        macro_path = Path(cfg.biotech_nbi_macro_duration_parquet)
    else:
        macro_path = Path(cfg.biotech_macro_duration_parquet)
    if not macro_path.exists():
        raise FileNotFoundError(
            f"[ERR] macro parquet missing for panel_kind={cfg.panel_kind}: "
            f"{macro_path}"
        )
    df = pd.read_parquet(macro_path)
    for c in cols:
        if c not in df.columns:
            raise KeyError(
                f"[ERR] macro column '{c}' not in {macro_path}; "
                f"available cols: {sorted(df.columns.tolist())[:10]}..."
            )
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    panel_index = pd.DatetimeIndex(
        pd.to_datetime(bridge.dates).normalize()
    )
    aligned = df.reindex(panel_index).ffill(limit=5)
    out: Dict[str, np.ndarray] = {}
    for c in cols:
        arr = aligned[c].to_numpy(dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        out[c] = arr
    return out


def _per_step_p_hat(
    score_spread_topk: np.ndarray, calibrator
) -> np.ndarray:
    """Calibrated probability evaluated at each day's top-K spread."""
    spread = np.asarray(score_spread_topk, dtype=np.float64).ravel()
    if spread.size == 0:
        return np.zeros(0, dtype=np.float64)
    p = calibrator.predict_proba(spread).astype(np.float64).ravel()
    if p.shape[0] != spread.shape[0]:
        raise RuntimeError(
            "[ERR] calibrator.predict_proba length mismatch: "
            f"{p.shape[0]} vs {spread.shape[0]}"
        )
    return p


def _per_step_mu_sigma(
    prior: KellySizingPrior,
    score_spread_topk: np.ndarray,
    score_uncertainty: np.ndarray,
    wrapper_returns: np.ndarray,
    calibrator,
    mu_scale: float,
    vol_window_days: int = 21,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-day ``mu_hat`` (daily-return units) and ``sigma_hat``.

    Mirrors the body of
    :func:`src.models.robust_invar_rl.prior_exposure.build_e_star_tape_from_aux`
    but returns the intermediate ``mu_hat`` and ``sigma_hat`` arrays
    for the compact obs builder. Keeps both calls in lockstep so the
    Phase 3 obs sees exactly the prior the residual SAC is anchoring.
    """
    spread = np.asarray(score_spread_topk, dtype=np.float64).ravel()
    unc = np.asarray(score_uncertainty, dtype=np.float64).ravel()
    rets = np.asarray(wrapper_returns, dtype=np.float64).ravel()
    T = spread.shape[0]
    if unc.shape[0] != T or rets.shape[0] != T:
        raise ValueError(
            "[ERR] _per_step_mu_sigma aux length mismatch: "
            f"spread={T} unc={unc.shape[0]} rets={rets.shape[0]}"
        )
    cfg = prior.cfg
    if T == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    if vol_window_days < 2:
        raise ValueError(
            f"[ERR] vol_window_days must be >= 2; got {vol_window_days}"
        )
    spread_scale = float(spread.std(ddof=1)) if T >= 2 else 0.0
    mu_scale_eff = float(max(float(mu_scale), cfg.mu_scale_floor))
    mu_out = np.zeros(T, dtype=np.float64)
    sigma_out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        if spread_scale > 1.0e-12:
            spread_norm = float(
                np.clip(spread[t] / spread_scale, -1.0, 1.0)
            )
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
        mu_out[t] = float(mu_signal * mu_scale_eff)
        if t == 0:
            sigma_out[t] = 0.0
        else:
            lo = max(0, t - vol_window_days + 1)
            sigma_out[t] = prior.compute_sigma_hat(rets[lo:t + 1])
    return mu_out, sigma_out


def _build_compact_obs_tape(
    prior: KellySizingPrior,
    aux,
    tape,
    e_star: np.ndarray,
    calibrator,
    mu_scale: float,
    vix_per_day: np.ndarray,
    ust10y_per_day: np.ndarray,
    regime_one_hot: Optional[np.ndarray],
    vol_window_days: int = 21,
) -> CompactObservationTape:
    """Assemble the Phase 3 :class:`CompactObservationTape` for one segment."""
    p_hat = _per_step_p_hat(aux.score_spread_topk, calibrator)
    mu_hat, sigma_hat = _per_step_mu_sigma(
        prior=prior,
        score_spread_topk=aux.score_spread_topk,
        score_uncertainty=aux.score_uncertainty,
        wrapper_returns=tape.base_return,
        calibrator=calibrator,
        mu_scale=mu_scale,
        vol_window_days=vol_window_days,
    )
    # Days alignment: aux.days and tape.days are identical by
    # construction (phase2_precompute mirrors tape day indices).
    T = int(tape.days.shape[0])
    if vix_per_day.shape[0] < int(tape.days.max()) + 1:
        raise ValueError(
            "[ERR] vix_per_day too short to index tape days; "
            f"got {vix_per_day.shape[0]} days vs max tape day "
            f"{int(tape.days.max())}"
        )
    if ust10y_per_day.shape[0] < int(tape.days.max()) + 1:
        raise ValueError(
            "[ERR] ust10y_per_day too short to index tape days; "
            f"got {ust10y_per_day.shape[0]} days vs max tape day "
            f"{int(tape.days.max())}"
        )
    vix_at_tape = vix_per_day[tape.days.astype(int)].astype(np.float64)
    ust10y_at_tape = ust10y_per_day[tape.days.astype(int)].astype(
        np.float64
    )
    if p_hat.shape[0] != T:
        raise RuntimeError(
            f"[ERR] p_hat length {p_hat.shape[0]} != T={T}"
        )
    return CompactObservationTape(
        p_hat=p_hat,
        mu_hat=mu_hat,
        sigma_hat=sigma_hat,
        e_star=np.asarray(e_star, dtype=np.float64).ravel(),
        vix_per_day=vix_at_tape,
        ust10y_per_day=ust10y_at_tape,
        regime_one_hot=regime_one_hot,
    )


def _load_day_to_cluster(universe: str, fold: int) -> Dict[int, int]:
    """Load argmax-cluster lookup from the k-means-8 regime probs cache.

    Returns a dict ``day_idx -> cluster_id``. The cache is the same one
    consumed by InVAR-RL-SIA. The driver is responsible for handling
    missing days gracefully (silent miss is allowed; the obs builder
    encodes those rows as all-zero, and the per-tape miss rate is
    logged for audit).
    """
    from invar_rl.layer2_sia.regime_probs import load_probs_lookup

    lookup = load_probs_lookup(universe, fold)
    out: Dict[int, int] = {}
    for d, probs in lookup.items():
        out[int(d)] = int(np.argmax(probs))
    return out


def _evaluate_episode(
    env: KellyPriorEnvWrapper,
) -> Dict[str, list]:
    """Run the test episode with residual_action = 0 (prior alone)."""
    obs, _ = env.reset(seed=0)
    info: Dict = {}
    rets: List[float] = []
    exps: List[float] = []
    e_stars: List[float] = []
    residuals: List[float] = []
    rewards: List[float] = []
    while True:
        action = np.asarray([0.0], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        rets.append(float(info["strategy_return"]))
        exps.append(float(info["e_final"]))
        e_stars.append(float(info["e_star"]))
        residuals.append(float(info["residual_action"]))
        rewards.append(float(reward))
        if term or trunc:
            break
    return {
        "strategy_returns": rets,
        "exposures": exps,
        "e_stars": e_stars,
        "residuals": residuals,
        "rewards": rewards,
    }


def _evaluate_with_agent(
    env: KellyPriorEnvWrapper, agent
) -> Dict[str, list]:
    obs, _ = env.reset(seed=0)
    info: Dict = {}
    rets: List[float] = []
    exps: List[float] = []
    e_stars: List[float] = []
    residuals: List[float] = []
    rewards: List[float] = []
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, term, trunc, info = env.step(action)
        rets.append(float(info["strategy_return"]))
        exps.append(float(info["e_final"]))
        e_stars.append(float(info["e_star"]))
        residuals.append(float(info["residual_action"]))
        rewards.append(float(reward))
        if term or trunc:
            break
    return {
        "strategy_returns": rets,
        "exposures": exps,
        "e_stars": e_stars,
        "residuals": residuals,
        "rewards": rewards,
    }


def _summary_stats(trace: Dict[str, list]) -> Dict[str, float]:
    arr = np.asarray(trace["strategy_returns"], dtype=np.float64)
    rew = np.asarray(trace["rewards"], dtype=np.float64)
    exp = np.asarray(trace["exposures"], dtype=np.float64)
    res = np.asarray(trace["residuals"], dtype=np.float64)
    return {
        "mean_reward": float(rew.mean()) if rew.size else 0.0,
        "mean_return": float(arr.mean()) if arr.size else 0.0,
        "volatility": float(arr.std()) if arr.size else 0.0,
        "n_steps": int(arr.size),
        "exposure_min": float(exp.min()) if exp.size else 0.0,
        "exposure_max": float(exp.max()) if exp.size else 0.0,
        "exposure_mean": float(exp.mean()) if exp.size else 0.0,
        "residual_abs_mean": float(np.abs(res).mean()) if res.size else 0.0,
    }


def _persist_daily_tape(
    test_tape, bridge, trace: Dict[str, list], out_path: Path,
) -> None:
    """Persist per-test-day daily tape parquet (Phase 2 schema)."""
    rets = np.asarray(trace["strategy_returns"], dtype=np.float64)
    exps = np.asarray(trace["exposures"], dtype=np.float64)
    e_stars = np.asarray(trace["e_stars"], dtype=np.float64)
    residuals = np.asarray(trace["residuals"], dtype=np.float64)
    n = int(
        min(
            test_tape.days.shape[0],
            rets.shape[0],
            exps.shape[0],
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "date",
        "exposure",
        "e_star",
        "residual_action",
        "strategy_return",
        "base_return",
    ]
    if n == 0:
        pd.DataFrame(columns=cols).to_parquet(out_path, index=False)
        print(f"{_LOG_PREFIX} [INFO] daily tape (empty) -> {out_path}", flush=True)
        return
    dates = pd.to_datetime(
        [str(bridge.dates[int(d)]) for d in test_tape.days[:n]]
    ).normalize()
    df = pd.DataFrame(
        {
            "date": dates,
            "exposure": exps[:n],
            "e_star": e_stars[:n],
            "residual_action": residuals[:n],
            "strategy_return": rets[:n],
            "base_return": test_tape.base_return[:n].astype(np.float64),
        }
    )
    df.to_parquet(out_path, index=False)
    print(
        f"{_LOG_PREFIX} [INFO] daily tape (n={n}) -> {out_path}",
        flush=True,
    )


def _resolve_layer1_ckpt(
    ckpt: Optional[str], ckpt_root: Optional[str], fold: int, seed: int
) -> Path:
    """Resolve the L1 ckpt path; supports explicit file or root-dir."""
    if ckpt:
        p = Path(ckpt)
        if not p.exists():
            raise FileNotFoundError(
                f"[ERR] --layer1-ckpt path does not exist: {p}"
            )
        return p
    if ckpt_root:
        root = Path(ckpt_root)
        candidate = root / f"fold{fold}_seed{seed}_full.pt"
        if not candidate.exists():
            raise FileNotFoundError(
                f"[ERR] no L1 ckpt at {candidate} under root {root}"
            )
        return candidate
    raise ValueError(
        "[ERR] supply --layer1-ckpt or --layer1-ckpt-root"
    )


def run_one_cell(
    fold: int,
    seed: int,
    ckpt_path: Path,
    layer2_yaml: Path,
    layer3_yaml: Path,
    stage3_yaml: Path,
    output_dir: Path,
    ckpt_out_dir: Path,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
    device: torch.device,
    equal_topk_k: int,
    delta_cap: float,
    kappa: float,
    e_max: float,
    calibration_method: str,
    daily_tape_dir: Optional[Path],
    mode: str,
    long_only: bool = False,
    phase3_compact_obs: bool = False,
    phase3_online_sharpe_reward: bool = False,
    phase3_regime_one_hot: bool = True,
    phase3_universe_label: str = "sp500",
    phase3_sharpe_half_life: int = 21,
    phase3_sharpe_warmup_steps: int = 5,
    phase3_sharpe_clip: float = 8.0,
    phase3_vix_scale: float = 20.0,
    phase3_ust10y_scale: float = 4.0,
) -> Dict:
    set_global_seed(seed)
    layer2 = load_layer2_config(str(layer2_yaml))
    layer3 = load_layer3_config(str(layer3_yaml))
    stage3 = load_stage3_config(str(stage3_yaml))

    cfg = InVARConfig(fold=fold, seed=seed)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    bridge = build_lattice_bridge(cfg)
    bundle = load_trained_invar(
        ckpt_path=ckpt_path, bridge=bridge, device=device
    )

    print(
        f"{_LOG_PREFIX} fold={fold} seed={seed} "
        f"K={equal_topk_k} delta_cap={delta_cap} "
        f"kappa={kappa} e_max={e_max} "
        f"calibration={calibration_method} mode={mode}",
        flush=True,
    )

    # 1. Wrapper tape on each segment (equal-weight top-K; no QP).
    train_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.train_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        score_mode="canonical",
        weighting_mode="equal_topk",
        ablation_seed=seed,
        long_only=long_only,
        equal_topk_k=int(equal_topk_k),
    )
    val_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.val_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        score_mode="canonical",
        weighting_mode="equal_topk",
        ablation_seed=seed + 1000,
        long_only=long_only,
        equal_topk_k=int(equal_topk_k),
    )
    test_tape = precompute_tape_canonical(
        bundle=bundle, bridge=bridge,
        day_indices=list(bridge.test_idx),
        layer2=layer2, stride=stage3.precompute_stride,
        score_mode="canonical",
        weighting_mode="equal_topk",
        ablation_seed=seed + 2000,
        long_only=long_only,
        equal_topk_k=int(equal_topk_k),
    )
    print(
        f"{_LOG_PREFIX} tapes: train={len(train_tape)} "
        f"val={len(val_tape)} test={len(test_tape)}",
        flush=True,
    )

    # 2. Aux per-day arrays (spread + uncertainty + profitable indicator).
    train_aux = compute_phase2_aux(
        bundle=bundle, bridge=bridge,
        tape_days=train_tape.days,
        base_return=train_tape.base_return,
        K=int(equal_topk_k),
    )
    val_aux = compute_phase2_aux(
        bundle=bundle, bridge=bridge,
        tape_days=val_tape.days,
        base_return=val_tape.base_return,
        K=int(equal_topk_k),
    )
    test_aux = compute_phase2_aux(
        bundle=bundle, bridge=bridge,
        tape_days=test_tape.days,
        base_return=test_tape.base_return,
        K=int(equal_topk_k),
    )
    if val_aux.daily_profitable.size == 0:
        raise RuntimeError(
            f"[ERR] val_aux is empty; cannot fit calibrator (fold={fold} seed={seed})"
        )
    # 3. Fit calibrator on val.
    cal = build_calibrator(calibration_method)
    val_spread = val_aux.score_spread_topk
    val_label = val_aux.daily_profitable
    n_classes = int(np.unique(val_label).size)
    if n_classes < 2:
        raise RuntimeError(
            "[ERR] val segment has a single-class profitable indicator "
            f"(unique={np.unique(val_label).tolist()}); calibrator "
            "needs both classes."
        )
    cal.fit(val_spread, val_label.astype(np.int64))
    print(
        f"{_LOG_PREFIX} calibrator fit: method={calibration_method} "
        f"n_val={val_label.size} positive_frac={val_label.mean():.3f}",
        flush=True,
    )

    # 4. Build per-segment e_star tapes.
    prior_cfg = KellySizingPriorConfig(
        kappa=float(kappa), e_max=float(e_max),
    )
    prior = KellySizingPrior(prior_cfg)
    # Cross-segment mu_scale: typical |daily strategy return| derived
    # from the validation segment (same window the calibrator was fit
    # on). Using a single per-tape scale ensures train/val/test all
    # share the same dimensionless-to-return conversion, so e_star is
    # comparable across segments.
    val_abs_ret = np.abs(
        np.asarray(val_tape.base_return, dtype=np.float64)
    )
    mu_scale_val = float(val_abs_ret.mean()) if val_abs_ret.size else 0.0
    mu_scale_val = float(max(mu_scale_val, prior_cfg.mu_scale_floor))
    print(
        f"{_LOG_PREFIX} mu_scale (val mean |base_return|): "
        f"{mu_scale_val:.6f}",
        flush=True,
    )
    train_estar = build_e_star_tape_from_aux(
        prior=prior,
        score_spread_topk=train_aux.score_spread_topk,
        score_uncertainty=train_aux.score_uncertainty,
        wrapper_returns=train_tape.base_return,
        calibrator=cal,
        mu_scale=mu_scale_val,
    )
    val_estar = build_e_star_tape_from_aux(
        prior=prior,
        score_spread_topk=val_aux.score_spread_topk,
        score_uncertainty=val_aux.score_uncertainty,
        wrapper_returns=val_tape.base_return,
        calibrator=cal,
        mu_scale=mu_scale_val,
    )
    test_estar = build_e_star_tape_from_aux(
        prior=prior,
        score_spread_topk=test_aux.score_spread_topk,
        score_uncertainty=test_aux.score_uncertainty,
        wrapper_returns=test_tape.base_return,
        calibrator=cal,
        mu_scale=mu_scale_val,
    )
    print(
        f"{_LOG_PREFIX} e_star summary: "
        f"train(mean={train_estar.mean():.3f}, std={train_estar.std():.3f}, "
        f"min={train_estar.min():.3f}, max={train_estar.max():.3f}) "
        f"val(mean={val_estar.mean():.3f}, std={val_estar.std():.3f}, "
        f"min={val_estar.min():.3f}, max={val_estar.max():.3f}) "
        f"test(mean={test_estar.mean():.3f}, std={test_estar.std():.3f}, "
        f"min={test_estar.min():.3f}, max={test_estar.max():.3f})",
        flush=True,
    )
    if not np.isfinite(train_estar).all():
        raise RuntimeError("[ERR] train e_star tape contains NaN/inf")
    if not np.isfinite(val_estar).all():
        raise RuntimeError("[ERR] val e_star tape contains NaN/inf")
    if not np.isfinite(test_estar).all():
        raise RuntimeError("[ERR] test e_star tape contains NaN/inf")

    # 5. Build wrapped envs.
    from stable_baselines3.common.monitor import Monitor

    ckpt_out_dir.mkdir(parents=True, exist_ok=True)
    curve_dir = ckpt_out_dir / f"sac_curves_f{fold}_s{seed}"
    curve_dir.mkdir(parents=True, exist_ok=True)

    train_inner = ExposureEnv(
        train_tape, layer3, bootstrap_episode=True
    )
    train_wrapped = KellyPriorEnvWrapper(
        inner_env=train_inner,
        e_star_tape=train_estar,
        delta_cap=float(delta_cap),
        e_max=float(layer3.exposure_max),
        e_min=float(layer3.exposure_min),
    )

    test_inner = ExposureEnv(
        test_tape, layer3, bootstrap_episode=False
    )
    test_wrapped = KellyPriorEnvWrapper(
        inner_env=test_inner,
        e_star_tape=test_estar,
        delta_cap=float(delta_cap),
        e_max=float(layer3.exposure_max),
        e_min=float(layer3.exposure_min),
    )

    # Phase 3 outer wrappers (compact obs + online Sharpe reward).
    # Backward-compatible: when both flags are off the env stack is
    # identical to Phase 2's.
    phase3_diag: Dict = {
        "enabled_compact_obs": bool(phase3_compact_obs),
        "enabled_online_sharpe_reward": bool(phase3_online_sharpe_reward),
        "enabled_regime_one_hot": False,
        "compact_obs_dim_expected": 0,
        "compact_obs_dim_actual": 0,
        "regime_miss_rate_train": 0.0,
        "regime_miss_rate_val": 0.0,
        "regime_miss_rate_test": 0.0,
    }
    if phase3_compact_obs or phase3_online_sharpe_reward:
        raw_macros = _load_raw_macro_per_day(cfg=cfg, bridge=bridge)
        vix_per_day = raw_macros[_PHASE3_VIX_COL]
        ust10y_per_day = raw_macros[_PHASE3_UST10Y_COL]
        print(
            f"{_LOG_PREFIX} [INFO] phase3 raw macro: "
            f"vix(mean={vix_per_day.mean():.2f}, "
            f"max={vix_per_day.max():.2f}) "
            f"ust10y(mean={ust10y_per_day.mean():.2f}, "
            f"max={ust10y_per_day.max():.2f})",
            flush=True,
        )
        if phase3_compact_obs:
            day_to_cluster: Optional[Dict[int, int]] = None
            if phase3_regime_one_hot:
                try:
                    day_to_cluster = _load_day_to_cluster(
                        universe=phase3_universe_label, fold=int(fold)
                    )
                    phase3_diag["enabled_regime_one_hot"] = True
                    print(
                        f"{_LOG_PREFIX} [INFO] phase3 regime cache: "
                        f"{len(day_to_cluster)} days "
                        f"(universe={phase3_universe_label} fold={fold})",
                        flush=True,
                    )
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        "[ERR] phase3_regime_one_hot=True but k-means-8 "
                        "regime cache is missing; either disable "
                        "regime_one_hot or run "
                        "invar_rl/layer2_sia/regime_probs.precompute_all "
                        f"for universe={phase3_universe_label} fold={fold}. "
                        f"Inner error: {exc}"
                    ) from exc
            compact_cfg = CompactObservationConfig(
                include_regime_one_hot=bool(phase3_regime_one_hot),
                vix_scale=float(phase3_vix_scale),
                ust10y_scale=float(phase3_ust10y_scale),
            )
            # Build per-segment compact obs tapes.
            def _segment_compact_tape(
                aux, tape, estar, segment_name: str,
            ) -> CompactObservationTape:
                regime_oh: Optional[np.ndarray] = None
                if day_to_cluster is not None:
                    regime_oh = build_regime_one_hot(
                        tape_days=tape.days,
                        day_to_cluster=day_to_cluster,
                        n_clusters=N_REGIME_CLUSTERS,
                    )
                    miss = float(
                        (regime_oh.sum(axis=1) == 0.0).mean()
                    )
                    phase3_diag[
                        f"regime_miss_rate_{segment_name}"
                    ] = miss
                    if miss > 0.0:
                        print(
                            f"{_LOG_PREFIX} [WARN] regime miss-rate on "
                            f"{segment_name}: {miss:.3f}",
                            flush=True,
                        )
                return _build_compact_obs_tape(
                    prior=prior,
                    aux=aux,
                    tape=tape,
                    e_star=estar,
                    calibrator=cal,
                    mu_scale=mu_scale_val,
                    vix_per_day=vix_per_day,
                    ust10y_per_day=ust10y_per_day,
                    regime_one_hot=regime_oh,
                )

            train_compact = _segment_compact_tape(
                train_aux, train_tape, train_estar, "train"
            )
            test_compact = _segment_compact_tape(
                test_aux, test_tape, test_estar, "test"
            )
            train_wrapped = CompactObservationWrapper(
                train_wrapped, tape=train_compact, cfg=compact_cfg,
            )
            test_wrapped = CompactObservationWrapper(
                test_wrapped, tape=test_compact, cfg=compact_cfg,
            )
            actual_dim = int(train_wrapped.obs_dim)
            expected_dim = int(N_BASE_FIELDS) + (
                int(N_REGIME_CLUSTERS) if phase3_regime_one_hot else 0
            ) + 1
            phase3_diag["compact_obs_dim_actual"] = actual_dim
            phase3_diag["compact_obs_dim_expected"] = expected_dim
            if actual_dim != expected_dim:
                raise RuntimeError(
                    "[ERR] compact obs dim mismatch: actual="
                    f"{actual_dim} expected={expected_dim}"
                )
            print(
                f"{_LOG_PREFIX} [INFO] phase3 compact obs dim = "
                f"{actual_dim} (expected={expected_dim}; "
                f"regime_one_hot={phase3_regime_one_hot})",
                flush=True,
            )
        if phase3_online_sharpe_reward:
            sharpe_cfg = OnlineSharpeRewardConfig(
                half_life_days=int(phase3_sharpe_half_life),
                warmup_steps=int(phase3_sharpe_warmup_steps),
                clip=float(phase3_sharpe_clip),
            )
            train_wrapped = OnlineSharpeRewardWrapper(
                train_wrapped, cfg=sharpe_cfg,
            )
            test_wrapped = OnlineSharpeRewardWrapper(
                test_wrapped, cfg=sharpe_cfg,
            )
            print(
                f"{_LOG_PREFIX} [INFO] phase3 online Sharpe reward: "
                f"half_life={phase3_sharpe_half_life} "
                f"warmup={phase3_sharpe_warmup_steps} "
                f"clip={phase3_sharpe_clip}",
                flush=True,
            )

    train_env = Monitor(train_wrapped, filename=str(curve_dir / "monitor"))

    # 6. Train + evaluate.
    if mode == "e_star_only":
        trace = _evaluate_episode(test_wrapped)
        agent_path = None
    elif mode == "e_star_plus_sac":
        from stable_baselines3 import SAC

        agent = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=stage3.learning_rate,
            seed=seed,
            verbose=0,
        )
        agent.learn(total_timesteps=stage3.total_timesteps)
        agent_path = ckpt_out_dir / f"sac_f{fold}_s{seed}.zip"
        agent.save(str(agent_path))
        trace = _evaluate_with_agent(test_wrapped, agent)
    else:
        raise ValueError(
            f"[ERR] unknown --mode {mode!r}; "
            "expected 'e_star_only' or 'e_star_plus_sac'"
        )

    stats = _summary_stats(trace)

    # Residual-cap respected check.
    exp_arr = np.asarray(trace["exposures"], dtype=np.float64)
    cap_violations = int(
        ((exp_arr < float(layer3.exposure_min) - 1.0e-9)
         | (exp_arr > float(layer3.exposure_max) + 1.0e-9)).sum()
    )
    if cap_violations > 0:
        print(
            f"{_LOG_PREFIX} [WARN] {cap_violations} cap violations in test",
            flush=True,
        )

    if daily_tape_dir is not None:
        tape_path = daily_tape_dir / "sac" / f"fold{fold}_seed{seed}.parquet"
        _persist_daily_tape(
            test_tape=test_tape, bridge=bridge, trace=trace, out_path=tape_path,
        )

    phase3_active = bool(
        phase3_compact_obs or phase3_online_sharpe_reward
    )
    if phase3_active:
        model_label = (
            "Robust-InVAR-RL Phase 3: residual SAC + Kelly prior "
            f"(K={equal_topk_k} delta_cap={delta_cap} "
            f"kappa={kappa} e_max={e_max} "
            f"calibration={calibration_method} "
            f"compact_obs={phase3_compact_obs} "
            f"online_sharpe={phase3_online_sharpe_reward})"
        )
    else:
        model_label = (
            "Robust-InVAR-RL Phase 2: residual SAC + Kelly prior "
            f"(K={equal_topk_k} delta_cap={delta_cap} "
            f"kappa={kappa} e_max={e_max} "
            f"calibration={calibration_method})"
        )
    payload = {
        "fold": fold,
        "seed": seed,
        "mode": mode,
        "model": model_label,
        "n_train_steps": len(train_tape),
        "n_val_steps": len(val_tape),
        "n_test_steps": len(test_tape),
        "methods": {"sac": stats},
        "diagnostics": {
            "cap_violations": cap_violations,
            "mu_scale_val": float(mu_scale_val),
            "train_e_star_mean": float(train_estar.mean()),
            "train_e_star_std": float(train_estar.std()),
            "val_e_star_mean": float(val_estar.mean()),
            "val_e_star_std": float(val_estar.std()),
            "test_e_star_mean": float(test_estar.mean()),
            "test_e_star_std": float(test_estar.std()),
            "test_e_star_min": float(test_estar.min()),
            "test_e_star_max": float(test_estar.max()),
            "phase3": phase3_diag,
        },
        "config": {
            "panel_kind": panel_kind,
            "two_regime_val": two_regime_val,
            "panel_end": panel_end,
            "precompute_stride": stage3.precompute_stride,
            "total_timesteps": stage3.total_timesteps,
            "equal_topk_k": int(equal_topk_k),
            "delta_cap": float(delta_cap),
            "kappa": float(kappa),
            "e_max": float(e_max),
            "calibration_method": calibration_method,
            "long_only": bool(long_only),
            "phase3_compact_obs": bool(phase3_compact_obs),
            "phase3_online_sharpe_reward": bool(
                phase3_online_sharpe_reward
            ),
            "phase3_regime_one_hot": bool(phase3_regime_one_hot),
            "phase3_universe_label": str(phase3_universe_label),
            "phase3_sharpe_half_life": int(phase3_sharpe_half_life),
            "phase3_sharpe_warmup_steps": int(phase3_sharpe_warmup_steps),
            "phase3_sharpe_clip": float(phase3_sharpe_clip),
            "phase3_vix_scale": float(phase3_vix_scale),
            "phase3_ust10y_scale": float(phase3_ust10y_scale),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"{_LOG_PREFIX} wrote {out_path}", flush=True)
    print(
        f"{_LOG_PREFIX} sac: mean_return={stats['mean_return']:+.5f} "
        f"vol={stats['volatility']:.5f} n_steps={stats['n_steps']} "
        f"exposure[min/mean/max]={stats['exposure_min']:.3f}/"
        f"{stats['exposure_mean']:.3f}/{stats['exposure_max']:.3f}",
        flush=True,
    )
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Robust-InVAR-RL Phase 2: residual SAC + Kelly prior on SP500."
        )
    )
    p.add_argument(
        "--fold", type=int, required=True, choices=[1, 2, 3, 4, 5]
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--layer1-ckpt", type=str, default=None)
    p.add_argument(
        "--layer1-ckpt-root", type=str, default=None,
        help=(
            "Root dir containing fold{F}_seed{S}_full.pt; the driver "
            "resolves the per-cell L1 ckpt within."
        ),
    )
    p.add_argument(
        "--layer2", type=str, default="invar_rl/configs/layer2.yaml"
    )
    p.add_argument(
        "--layer3", type=str, default="invar_rl/configs/layer3.yaml"
    )
    p.add_argument(
        "--stage3", type=str, default="invar_rl/configs/stage3.yaml"
    )
    p.add_argument(
        "--output-dir-root",
        type=str,
        default="outputs/sp500/layer3_robust_phase2_25cell",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "If set, overrides --output-dir-root with an explicit "
            "destination directory for foldF_seedS.json."
        ),
    )
    p.add_argument(
        "--daily-tape-dir",
        type=str,
        default=None,
        help=(
            "Destination root for the daily-tape parquet "
            "(default = <output-dir>/daily_tape). Off when '--no-tape'."
        ),
    )
    p.add_argument("--no-tape", action="store_true", default=False)
    p.add_argument(
        "--panel_kind", type=str, default="lattice_native",
        choices=["biotech", "lattice_native"],
    )
    p.add_argument("--panel_end", type=str, default="2025-12-31")
    p.add_argument("--two_regime_val", action="store_true", default=True)
    p.add_argument(
        "--equal-topk-k", type=int, default=50,
        help="Per-side K for the wrapper (default 50 = SP500 canonical).",
    )
    p.add_argument(
        "--delta-cap", type=float, default=0.25,
        help="Max absolute residual the actor can add (exposure units).",
    )
    p.add_argument(
        "--kappa", type=float, default=0.05,
        help=(
            "Kelly fraction applied to mu_hat / sigma_hat^2. "
            "Default 0.05 is a 5%% fractional Kelly (full Kelly would "
            "saturate the e_max clip for any sensible signal)."
        ),
    )
    p.add_argument(
        "--e-max", type=float, default=1.5,
        help="Upper bound on the prior; passed through to the final clip.",
    )
    p.add_argument(
        "--calibration-method", type=str, default="platt",
        choices=["platt", "isotonic"],
    )
    p.add_argument(
        "--mode", type=str, default="e_star_plus_sac",
        choices=["e_star_only", "e_star_plus_sac"],
        help="'e_star_only' skips SAC training (smoke check).",
    )
    p.add_argument(
        "--long-only", action="store_true", default=False,
    )
    # Phase 3: compact obs + online Sharpe reward. Both default OFF so
    # existing Phase 2 sbatches behave identically.
    p.add_argument(
        "--phase3-compact-obs", action="store_true", default=False,
        help=(
            "Phase 3: replace the Kelly-prior wrapper observation with "
            "the 8-base + 8-regime + 1-e_star (17-dim) compact obs."
        ),
    )
    p.add_argument(
        "--phase3-online-sharpe-reward",
        action="store_true", default=False,
        help=(
            "Phase 3: replace the per-step PnL reward with the EWMA "
            "online Sharpe increment."
        ),
    )
    p.add_argument(
        "--phase3-no-regime-one-hot",
        action="store_true", default=False,
        help=(
            "Phase 3 toggle: drop the 8-dim k-means-8 regime one-hot "
            "from the compact obs (8-dim base + 1-dim e_star, total 9)."
        ),
    )
    p.add_argument(
        "--phase3-universe-label", type=str, default="sp500",
        help=(
            "Universe label used to locate the k-means-8 regime cache "
            "at cache/dr_rl/regime_probs/{universe}/fold{F}/probs.parquet"
        ),
    )
    p.add_argument(
        "--phase3-sharpe-half-life", type=int, default=21,
        help="EWMA half-life (days) for the online Sharpe reward.",
    )
    p.add_argument(
        "--phase3-sharpe-warmup-steps", type=int, default=5,
        help=(
            "Warm-up steps during which the online Sharpe reward "
            "returns r_t * sqrt(252) (avoids divide-by-tiny-sigma)."
        ),
    )
    p.add_argument(
        "--phase3-sharpe-clip", type=float, default=8.0,
        help="Absolute clip applied to the online Sharpe per-step reward.",
    )
    p.add_argument(
        "--phase3-vix-scale", type=float, default=20.0,
        help="Divisor applied to raw VIX in the compact obs.",
    )
    p.add_argument(
        "--phase3-ust10y-scale", type=float, default=4.0,
        help="Divisor applied to raw UST10Y (DGS10) in the compact obs.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ckpt_path = _resolve_layer1_ckpt(
        ckpt=args.layer1_ckpt, ckpt_root=args.layer1_ckpt_root,
        fold=int(args.fold), seed=int(args.seed),
    )

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.output_dir_root)
    ckpt_out_dir = out_dir / "_ckpt"
    if args.no_tape:
        tape_dir = None
    elif args.daily_tape_dir:
        tape_dir = Path(args.daily_tape_dir)
    else:
        tape_dir = out_dir / "daily_tape"

    run_one_cell(
        fold=int(args.fold),
        seed=int(args.seed),
        ckpt_path=ckpt_path,
        layer2_yaml=Path(args.layer2),
        layer3_yaml=Path(args.layer3),
        stage3_yaml=Path(args.stage3),
        output_dir=out_dir,
        ckpt_out_dir=ckpt_out_dir,
        panel_kind=args.panel_kind,
        panel_end=args.panel_end,
        two_regime_val=bool(args.two_regime_val),
        device=device,
        equal_topk_k=int(args.equal_topk_k),
        delta_cap=float(args.delta_cap),
        kappa=float(args.kappa),
        e_max=float(args.e_max),
        calibration_method=str(args.calibration_method),
        daily_tape_dir=tape_dir,
        mode=str(args.mode),
        long_only=bool(args.long_only),
        phase3_compact_obs=bool(args.phase3_compact_obs),
        phase3_online_sharpe_reward=bool(args.phase3_online_sharpe_reward),
        phase3_regime_one_hot=not bool(args.phase3_no_regime_one_hot),
        phase3_universe_label=str(args.phase3_universe_label),
        phase3_sharpe_half_life=int(args.phase3_sharpe_half_life),
        phase3_sharpe_warmup_steps=int(args.phase3_sharpe_warmup_steps),
        phase3_sharpe_clip=float(args.phase3_sharpe_clip),
        phase3_vix_scale=float(args.phase3_vix_scale),
        phase3_ust10y_scale=float(args.phase3_ust10y_scale),
    )


if __name__ == "__main__":
    main()
