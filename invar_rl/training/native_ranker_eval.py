"""Native eval for Layer-1 ranker baselines: top-decile long minus
bottom-decile short, equal-weight, daily rebalance.

This is the published native evaluation protocol for cross-sectional
return predictors (MASTER, FactorVAE, iTransformer, StockMixer, DySTAGE,
MERA, SWA-InVAR, and canonical InVAR). No MV-QP, no RL — just the
predictor's own scores rolled into a top-K long / bottom-K short
dollar-neutral portfolio on the test segment.

The metric (annualised Sharpe on the test log-return series) is exactly
the same as the one InVAR-RL uses, so the comparison in the paper is
fair: every method's pipeline ends with a daily log-return on the
identical test segments, and the difference is attributable to the
method's choices.

Pulls predictions from
``results/baselines_universal_two_regime_val/{baseline}/foldF_seedS_predictions.npz``.

Output: ``invar_rl/results/native_ranker_baselines/{baseline}/foldF_seedS.json``.
Schema matches the stage3 RL output (mean_return / volatility /
sharpe / final_equity) for direct table merge.

Usage::

    python -m invar_rl.training.native_ranker_eval \
        --baseline master --fold 1 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.invar import InVARConfig

from invar_rl.data.lattice_bridge import build_lattice_bridge


_RANKER_BASELINES = (
    "master", "factorvae", "itransformer", "stockmixer",
    "dystage", "mera", "swa_invar",
)


def _topk_ls_portfolio(
    y_hat: np.ndarray,
    tradable: np.ndarray,
    log_returns: np.ndarray,
    day_indices,
    k: int = 50,
    gross: float = 1.0,
) -> dict:
    """Equal-weight top-k long / bottom-k short, daily rebalance.

    Args:
        y_hat: (T, N) per-day predicted scores from the baseline.
        tradable: (T, N) bool mask of tradeable stocks per day.
        log_returns: (T, N) realised 1-day log return per stock.
        day_indices: trading-day indices to evaluate (test segment).
        k: number of long positions = number of short positions.
        gross: total L1 of weights (gross = 1.0 -> 50% long + 50% short).

    Returns:
        Per-day log-return series + summary metrics.
    """
    daily = []
    gross_hist = []
    net_hist = []
    n_long = []
    n_short = []
    per_name = gross / (2.0 * k)
    for d in day_indices:
        if d + 1 >= log_returns.shape[0]:
            break
        active = np.nonzero(tradable[d])[0]
        if active.size < 2 * k:
            continue
        scores = y_hat[d, active].astype(np.float64)
        valid = np.isfinite(scores)
        if valid.sum() < 2 * k:
            continue
        scores = np.where(valid, scores, -np.inf)
        order = np.argsort(scores)
        short_local = order[:k]
        long_local = order[-k:]
        w = np.zeros(active.size, dtype=np.float64)
        w[long_local] = per_name
        w[short_local] = -per_name
        r_next = log_returns[d + 1, active]
        r_next = np.where(np.isfinite(r_next), r_next, 0.0)
        daily.append(float((w * r_next).sum()))
        gross_hist.append(float(np.sum(np.abs(w))))
        net_hist.append(float(np.sum(w)))
        n_long.append(k)
        n_short.append(k)
    daily = np.asarray(daily)
    daily = daily[np.isfinite(daily)]
    mean = float(daily.mean()) if daily.size else 0.0
    vol = float(daily.std(ddof=1)) if daily.size > 1 else 0.0
    ann_ret = mean * 252.0
    ann_vol = vol * np.sqrt(252.0)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    return {
        "mean_return": mean,
        "volatility": vol,
        "sharpe_annualised": sharpe,
        "final_equity": float(np.exp(daily.sum())) if daily.size else 1.0,
        "n_steps": int(daily.size),
        "gross_exposure_mean": (
            float(np.mean(gross_hist)) if gross_hist else 0.0
        ),
        "net_exposure_mean": (
            float(np.mean(net_hist)) if net_hist else 0.0
        ),
        "k_long": k,
        "k_short": k,
        # Per-day log-return series for the daily cumulative-return figure
        # (Figure 8); summary stats above are recomputable from this.
        "daily_log_returns": [float(x) for x in daily.tolist()],
    }


def run_one_cell(
    baseline: str,
    fold: int,
    seed: int,
    npz_root: Path,
    output_dir: Path,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
    k_values=(25, 50, 100),
) -> dict:
    cfg = InVARConfig(fold=fold, seed=seed)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    bridge = build_lattice_bridge(cfg)

    npz_path = (
        Path(npz_root) / baseline
        / f"fold{fold}_seed{seed}_predictions.npz"
    )
    if not npz_path.exists():
        raise FileNotFoundError(
            f"baseline predictions not found: {npz_path}"
        )
    blob = np.load(npz_path, allow_pickle=False)
    y_hat = blob["y_hat"]
    tradable = blob["tradable_mask"]
    log_returns = bridge.log_returns_1d

    if y_hat.shape != log_returns.shape:
        raise ValueError(
            f"baseline {baseline} y_hat shape {y_hat.shape} "
            f"!= bridge log_returns shape {log_returns.shape}"
        )

    methods = {}
    for k in k_values:
        res = _topk_ls_portfolio(
            y_hat=y_hat,
            tradable=tradable,
            log_returns=log_returns,
            day_indices=list(bridge.test_idx),
            k=k,
            gross=1.0,
        )
        method_name = f"topk_ls_k{k}"
        methods[method_name] = res
        print(
            f"  {method_name:14s} sharpe={res['sharpe_annualised']:+.3f} "
            f"ann_ret={res['mean_return']*252:+.4f} "
            f"ann_vol={res['volatility']*(252**0.5):+.4f} "
            f"eq={res['final_equity']:.4f}"
        )

    payload = {
        "baseline": baseline,
        "fold": fold,
        "seed": seed,
        "model": (
            f"Native ranker baseline L1={baseline} -> "
            f"top-K L/S equal-weight daily rebalance "
            f"(published native eval; no RL, no MV-QP)"
        ),
        "n_test_days": int(len(bridge.test_idx)),
        "methods": methods,
        "config": {
            "panel_kind": panel_kind,
            "two_regime_val": two_regime_val,
            "panel_end": panel_end,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[native ranker eval] wrote {out_path}")
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Native top-K L/S eval for Layer-1 ranker baselines."
    )
    p.add_argument(
        "--baseline", type=str, required=True,
        choices=list(_RANKER_BASELINES),
    )
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--npz-root", type=str,
        default="results/baselines_universal_two_regime_val",
    )
    p.add_argument(
        "--output-dir-root", type=str,
        default="invar_rl/results/native_ranker_baselines",
    )
    p.add_argument(
        "--panel_kind", type=str, default="lattice_native",
        choices=["biotech", "lattice_native"],
    )
    p.add_argument("--panel_end", type=str, default="2025-12-31")
    p.add_argument("--two_regime_val", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir_root) / args.baseline
    print(
        f"[native ranker eval] baseline={args.baseline} "
        f"fold={args.fold} seed={args.seed}"
    )
    run_one_cell(
        baseline=args.baseline,
        fold=args.fold,
        seed=args.seed,
        npz_root=Path(args.npz_root),
        output_dir=out_dir,
        panel_kind=args.panel_kind,
        panel_end=args.panel_end,
        two_regime_val=args.two_regime_val,
    )


if __name__ == "__main__":
    main()
