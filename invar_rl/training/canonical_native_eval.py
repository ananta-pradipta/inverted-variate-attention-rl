"""Daily top-K L/S evaluation of canonical InVAR scores.

Mirrors invar_rl/training/native_ranker_eval.py but the score source
is the canonical InVAR full-state ckpt rather than an external
baseline's saved y_hat. Produces a per-(fold, seed) JSON with the
same top-K L/S Sharpe schema so canonical InVAR can be placed in the
paper's Panel A long-short ranker-baseline comparison row.

Usage::

    python -m invar_rl.training.canonical_native_eval \
        --fold 1 --seed 42 \
        --layer1-ckpt invar_rl/results/stage1/_ckpt/fold1_seed42_full.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.invar import InVARConfig

from invar_rl.data.lattice_bridge import build_lattice_bridge
from invar_rl.layer1_ranker.canonical_runner import load_trained_invar


def _topk_ls_portfolio(
    scores_by_day: dict,  # day_idx -> (active_global, scores) tuple
    tradable: np.ndarray,
    log_returns: np.ndarray,
    day_indices,
    k: int = 25,
) -> dict:
    daily = []
    for d in day_indices:
        if d + 1 >= log_returns.shape[0]:
            break
        if d not in scores_by_day:
            continue
        active, scores = scores_by_day[d]
        if active.size < 2 * k:
            continue
        valid = np.isfinite(scores)
        if valid.sum() < 2 * k:
            continue
        masked = np.where(valid, scores, -np.inf)
        order = np.argsort(masked)
        short_local = order[:k]
        long_local = order[-k:]
        per_name = 1.0 / (2.0 * k)
        w = np.zeros(active.size, dtype=np.float64)
        w[long_local] = per_name
        w[short_local] = -per_name
        r_next = log_returns[d + 1, active]
        r_next = np.where(np.isfinite(r_next), r_next, 0.0)
        daily.append(float((w * r_next).sum()))
    arr = np.asarray(daily, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    mean = float(arr.mean()) if arr.size else 0.0
    vol = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ann_ret = mean * 252.0
    ann_vol = vol * np.sqrt(252.0)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    return {
        "mean_return": mean,
        "volatility": vol,
        "sharpe_annualised": sharpe,
        "final_equity": float(np.exp(arr.sum())) if arr.size else 1.0,
        "n_steps": int(arr.size),
        "k_long": k,
        "k_short": k,
    }


def run_one_cell(
    fold: int,
    seed: int,
    ckpt_path: Path,
    output_dir: Path,
    panel_kind: str = "lattice_native",
    panel_end: str = "2025-12-31",
    two_regime_val: bool = True,
    k_values=(25, 50, 100),
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = InVARConfig(fold=fold, seed=seed)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    bridge = build_lattice_bridge(cfg)
    bundle = load_trained_invar(
        ckpt_path=ckpt_path, bridge=bridge, device=device,
    )

    # Forward through every test day, cache scores.
    scores_by_day = {}
    for d in list(bridge.test_idx):
        try:
            out = bundle.forward_day(int(d))
        except (ValueError, RuntimeError):
            continue
        active = out["active_indices"].cpu().numpy().astype(np.int64)
        scores = out["scores"].detach().cpu().numpy().astype(np.float64)
        scores_by_day[int(d)] = (active, scores)

    methods = {}
    for k in k_values:
        res = _topk_ls_portfolio(
            scores_by_day=scores_by_day,
            tradable=bridge.tradable,
            log_returns=bridge.log_returns_1d,
            day_indices=list(bridge.test_idx),
            k=k,
        )
        method_name = f"topk_ls_k{k}"
        methods[method_name] = res
        print(
            f"  {method_name} sharpe={res['sharpe_annualised']:+.3f} "
            f"ann_ret={res['mean_return']*252:+.4f} eq={res['final_equity']:.4f}"
        )
    payload = {
        "baseline": "canonical_invar",
        "fold": fold,
        "seed": seed,
        "model": (
            "Canonical InVAR L1 -> top-K L/S equal-weight daily "
            "rebalance (Panel A row)"
        ),
        "n_test_days": int(len(bridge.test_idx)),
        "methods": methods,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[canonical_native_eval] wrote {out_path}")
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Canonical InVAR top-K L/S daily eval (Panel A)."
    )
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--layer1-ckpt", type=str, required=True)
    p.add_argument(
        "--output-dir", type=str,
        default="invar_rl/results/native_ranker_baselines/canonical_invar",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_one_cell(
        fold=args.fold, seed=args.seed,
        ckpt_path=Path(args.layer1_ckpt),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
