"""InVAR-RL Stage 2: portfolio evaluation through the frozen canonical layer 1.

v0 of stage 2 in the InVAR-RL build (locked 2026-05-19). Layer 1 is the
canonical InVAR trained in Stage 1 (``invar_rl/training/stage1_rank.py``
delegating to ``src.invar.train_invar``), frozen for this stage. The
layer-2 mean-variance QP runs every test day with the canonical scores
as expected returns and a trailing realised-return covariance; the
realised portfolio return is recorded. No gradients flow into layer 1
in v0; "decision-focused" fine-tuning of layer 1 via QP gradients is
deferred to stage 4 (joint).

Per-(fold, seed) output JSON is written to
``invar_rl/results/stage2_eval/foldF_seedS.json`` with a fixed schema:

```
{
  "fold": int, "seed": int,
  "n_test_days": int, "n_active_mean": float,
  "test_mean_return": float, "test_volatility": float,
  "test_annualised_sharpe": float, "test_cumulative_log_return": float,
  "test_gross_exposure_mean": float, "test_predicted_vol_mean": float,
  "config": {layer2 knobs}
}
```

Usage::

    python -m invar_rl.training.stage2_eval \
        --fold 1 --seed 42 \
        --layer1-ckpt invar_rl/results/stage1/_ckpt/fold1_seed42_full.pt \
        --layer2 invar_rl/configs/layer2.yaml \
        --output-dir invar_rl/results/stage2_eval

Or via ``invar_rl/scripts/wulver/invar_rl_stage2_eval.sbatch``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.invar import InVARConfig

from invar_rl.common.config import load_layer2_config
from invar_rl.data.lattice_bridge import (
    LatticePanelBatch,
    build_lattice_bridge,
)
from invar_rl.layer1_ranker.canonical_runner import (
    TrainedInVARBundle,
    load_trained_invar,
)
import cvxpy as cp

from invar_rl.layer2_alloc.covariance import estimate_covariance


def _config_for(
    fold: int,
    seed: int,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
) -> InVARConfig:
    cfg = InVARConfig(fold=fold, seed=seed)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    if cfg.enable_retrieval_bank:
        raise RuntimeError("canonical InVAR is bankless")
    return cfg


def _covariance_for_day(
    bridge: LatticePanelBatch,
    day_idx: int,
    active_global: np.ndarray,
    cov_lookback: int,
    estimator: str,
    factor_rank: int,
) -> np.ndarray:
    """Trailing realised one-day return covariance, train-fold-floor."""
    train_start = int(bridge.train_idx[0])
    lo = max(train_start, day_idx - cov_lookback)
    hi = day_idx
    window = bridge.log_returns_1d[lo:hi, :][:, active_global]
    window = np.where(np.isfinite(window), window, 0.0)
    if window.shape[0] < 2:
        n = active_global.shape[0]
        return np.eye(n, dtype=np.float64)
    return estimate_covariance(window, estimator, factor_rank)


def _solve_mvqp_eval(
    s: np.ndarray,
    sigma: np.ndarray,
    gamma: float,
    bound: float,
    gross: float,
    long_only: bool = False,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Eval-only mean-variance QP solved directly with cvxpy.

    Matches the layer-2 formulation from
    :class:`invar_rl.layer2_alloc.qp_layer.MeanVarianceQP` (dollar-neutral,
    per-name box, gross-leverage cap) but builds a fresh cp.Problem
    per day so the DPP-compilation cost of CvxpyLayer is avoided. No
    gradients are required at eval time.

    Args:
        s, sigma, gamma, bound, gross: standard MV-QP inputs.
        long_only: if True, drop the dollar-neutral constraint and
            replace with a long-only fully-invested constraint
            ($w \\ge 0$, $\\sum w = 1$). Used for the apples-to-apples
            comparison against FinRL / StockFormer.

    Returns:
        A pair ``(weights, summary)``. ``weights`` is None on solver
        failure; ``summary`` carries ``predicted_vol``,
        ``gross_exposure``, ``effective_positions``, ``net_exposure``.
    """
    n = int(s.shape[0])
    w = cp.Variable(n)
    objective = cp.Minimize(
        -s @ w + 0.5 * gamma * cp.quad_form(w, cp.psd_wrap(sigma))
    )
    if long_only:
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0,
            w <= bound,
        ]
    else:
        constraints = [
            cp.sum(w) == 0,
            cp.norm(w, 1) <= gross,
            w <= bound,
            w >= -bound,
        ]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.SCS, verbose=False)
    except Exception:
        return None, {
            "predicted_vol": float("nan"),
            "gross_exposure": float("nan"),
            "effective_positions": 0.0,
            "net_exposure": float("nan"),
        }
    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None, {
            "predicted_vol": float("nan"),
            "gross_exposure": float("nan"),
            "effective_positions": 0.0,
            "net_exposure": float("nan"),
        }
    w_np = np.asarray(w.value, dtype=np.float64)
    pred_vol = float(np.sqrt(max(0.0, w_np @ sigma @ w_np)))
    gross_exp = float(np.sum(np.abs(w_np)))
    net_exp = float(np.sum(w_np))
    eff_pos = float(np.sum(w_np ** 2) ** 2 /
                    max(np.sum(w_np ** 4), 1e-12))
    return w_np, {
        "predicted_vol": pred_vol,
        "gross_exposure": gross_exp,
        "effective_positions": eff_pos,
        "net_exposure": net_exp,
    }


def _portfolio_log_return(
    weights: torch.Tensor,
    bridge: LatticePanelBatch,
    day_idx: int,
    active_global: np.ndarray,
) -> float:
    """Realised one-day log return of the portfolio held at end of day."""
    if day_idx + 1 >= bridge.log_returns_1d.shape[0]:
        return float("nan")
    next_ret = bridge.log_returns_1d[day_idx + 1, active_global]
    next_ret = np.where(np.isfinite(next_ret), next_ret, 0.0)
    w = weights.detach().cpu().numpy().astype(np.float64)
    return float((w * next_ret).sum())


def run_one_cell(
    fold: int,
    seed: int,
    ckpt_path: Path,
    layer2_yaml: Path,
    output_dir: Path,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
    device: torch.device,
    long_only: bool = False,
) -> Dict:
    """Evaluate canonical InVAR -> Layer 2 QP on the fold's test segment."""
    layer2 = load_layer2_config(str(layer2_yaml))
    cfg = _config_for(
        fold=fold,
        seed=seed,
        panel_kind=panel_kind,
        panel_end=panel_end,
        two_regime_val=two_regime_val,
    )
    bridge = build_lattice_bridge(cfg)
    bundle = load_trained_invar(
        ckpt_path=ckpt_path, bridge=bridge, device=device
    )

    test_days = list(bridge.test_idx)
    daily_returns = []
    n_active = []
    gross = []
    pred_vol = []
    # Cap the QP cross-section to MAX_QP_NAMES by absolute InVAR score so
    # the per-day CvxpyLayer build stays well under memory + DPP-compile
    # budget. Layer 1 still scores the full active cross-section; this
    # only restricts which names enter the portfolio QP.
    MAX_QP_NAMES = 200

    for day in test_days:
        active_global = np.nonzero(bridge.tradable[day])[0]
        if active_global.size < 5:
            continue
        out = bundle.forward_day(day)
        all_scores = out["scores"].to(device)
        active_indices_t = out["active_indices"]
        # all_scores is indexed by active_indices_t, NOT active_global;
        # but they should match because the bridge default active_mask
        # is tradable[day]. Cross-check sizes.
        if all_scores.numel() != active_global.size:
            # If they diverge (e.g., shape filter mismatch), fall back
            # to the bundle's active_indices.
            active_global = active_indices_t.cpu().numpy().astype(
                np.int64
            )
        if active_global.size > MAX_QP_NAMES:
            score_abs = all_scores.detach().cpu().abs().numpy()
            keep_local = np.argsort(-score_abs)[:MAX_QP_NAMES]
            keep_local.sort()
            scores = all_scores[keep_local]
            active_global = active_global[keep_local]
        else:
            scores = all_scores
        sigma_np = _covariance_for_day(
            bridge,
            day,
            active_global,
            layer2.cov_lookback,
            layer2.estimator,
            layer2.factor_rank,
        )
        # Eval-only QP via direct cvxpy (no differentiable layer needed).
        s_np = scores.detach().cpu().numpy().astype(np.float64)
        w_np, summary = _solve_mvqp_eval(
            s_np,
            sigma_np,
            gamma=float(layer2.risk_aversion),
            bound=float(layer2.per_name_bound),
            gross=float(layer2.gross_leverage),
            long_only=long_only,
        )
        if w_np is None:
            continue
        weights = torch.from_numpy(w_np)
        r = _portfolio_log_return(weights, bridge, day, active_global)
        if not np.isfinite(r):
            continue
        daily_returns.append(r)
        n_active.append(int(active_global.size))
        gross.append(float(summary["gross_exposure"]))
        pred_vol.append(float(summary["predicted_vol"]))

    arr = np.asarray(daily_returns)
    if arr.size == 0:
        raise RuntimeError("no eligible test days; check fold + ckpt")

    mean_return = float(arr.mean())
    vol = float(arr.std(ddof=1))
    sharpe_annual = (
        mean_return / vol * np.sqrt(252.0) if vol > 0 else 0.0
    )
    cum_log_return = float(arr.sum())

    payload = {
        "fold": fold,
        "seed": seed,
        "model": "InVAR-RL stage2 eval (canonical InVAR frozen + QP)",
        "n_test_days": int(arr.size),
        "n_active_mean": float(np.mean(n_active)),
        "test_mean_return": mean_return,
        "test_volatility": vol,
        "test_annualised_sharpe": float(sharpe_annual),
        "test_cumulative_log_return": cum_log_return,
        "test_gross_exposure_mean": float(np.mean(gross)),
        "test_predicted_vol_mean": float(np.mean(pred_vol)),
        "config": {
            "cov_lookback": layer2.cov_lookback,
            "estimator": layer2.estimator,
            "factor_rank": layer2.factor_rank,
            "panel_kind": panel_kind,
            "two_regime_val": two_regime_val,
            "panel_end": panel_end,
        },
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fold{fold}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[InVAR-RL stage 2 eval] wrote {out_path}")
    print(
        f"  test n_days={arr.size} mean_return={mean_return:+.5f} "
        f"vol={vol:.5f} annual_sharpe={sharpe_annual:+.3f}"
    )
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="InVAR-RL stage 2 eval: canonical InVAR + QP portfolio."
    )
    p.add_argument(
        "--fold", type=int, required=True, choices=[1, 2, 3, 4, 5]
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--layer1-ckpt",
        type=str,
        required=True,
        help="Path to a foldF_seedS_full.pt blob from stage 1.",
    )
    p.add_argument(
        "--layer2",
        type=str,
        default="invar_rl/configs/layer2.yaml",
        help="Layer 2 config yaml.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="invar_rl/results/stage2_eval",
    )
    p.add_argument(
        "--panel_kind",
        type=str,
        default="lattice_native",
        choices=["biotech", "lattice_native"],
    )
    p.add_argument("--panel_end", type=str, default="2025-12-31")
    p.add_argument("--two_regime_val", action="store_true", default=True)
    p.add_argument(
        "--long-only", action="store_true", default=False,
        help="Use long-only fully-invested QP instead of dollar-neutral L/S.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(
        f"[InVAR-RL stage 2 eval] fold={args.fold} seed={args.seed} "
        f"ckpt={args.layer1_ckpt} device={device}"
    )
    run_one_cell(
        fold=args.fold,
        seed=args.seed,
        ckpt_path=Path(args.layer1_ckpt),
        layer2_yaml=Path(args.layer2),
        output_dir=Path(args.output_dir),
        panel_kind=args.panel_kind,
        panel_end=args.panel_end,
        two_regime_val=args.two_regime_val,
        device=device,
        long_only=args.long_only,
    )


if __name__ == "__main__":
    main()
