"""Run StockFormer-style and FinRL-style whole-stack baselines per cell.

For one (baseline, fold, seed) cell, builds the bridge, trains the
baseline end-to-end on the train segment, evaluates on the test
segment, writes a JSON in the same schema as stage3 results.

Usage::

    python -m invar_rl.training.whole_stack_rl_eval \
        --baseline finrl --fold 1 --seed 42
    python -m invar_rl.training.whole_stack_rl_eval \
        --baseline stockformer --fold 1 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.invar import InVARConfig

from invar_rl.baselines.whole_stack_rl import (
    run_finrl_baseline,
    run_stockformer_baseline,
)
from invar_rl.common.seeding import set_global_seed
from invar_rl.data.lattice_bridge import build_lattice_bridge


SUPPORTED = ("finrl", "stockformer")


def run_one_cell(
    baseline: str,
    fold: int,
    seed: int,
    output_dir: Path,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
    total_timesteps: int,
    universe_k: int,
    feature_set: str = "lite",
) -> dict:
    set_global_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = InVARConfig(fold=fold, seed=seed)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    bridge = build_lattice_bridge(cfg)

    if baseline == "finrl":
        res = run_finrl_baseline(
            bridge=bridge, fold=fold, seed=seed,
            total_timesteps=total_timesteps,
            universe_k=universe_k,
            device=device,
            feature_set=feature_set,
        )
    elif baseline == "stockformer":
        res = run_stockformer_baseline(
            bridge=bridge, fold=fold, seed=seed,
            total_timesteps=total_timesteps,
            universe_k=universe_k,
            device=device,
            feature_set=feature_set,
        )
    else:
        raise ValueError(f"unsupported baseline: {baseline}")

    d = res.as_dict()
    print(
        f"[whole_stack_rl_eval baseline={baseline}] "
        f"fold={fold} seed={seed} sharpe={d['sharpe_annualised']:+.3f} "
        f"ann_ret={d['mean_return']*252:+.4f} "
        f"ann_vol={d['volatility']*(252**0.5):+.4f} "
        f"eq={d['final_equity']:.4f}"
    )

    payload = {
        "baseline": baseline,
        "fold": fold,
        "seed": seed,
        "model": (
            f"Whole-stack RL baseline = {baseline} "
            f"(native re-implementation, end-to-end on lattice_native)"
        ),
        "n_test_days": int(d["n_steps"]),
        "methods": {res.name: d},
        "config": {
            "panel_kind": panel_kind,
            "two_regime_val": two_regime_val,
            "panel_end": panel_end,
            "total_timesteps": total_timesteps,
            "universe_k": universe_k,
            "device": device,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[whole_stack_rl_eval] wrote {out_path}")
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Whole-stack RL baselines (StockFormer / FinRL)."
    )
    p.add_argument(
        "--baseline", type=str, required=True, choices=list(SUPPORTED)
    )
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--output-dir-root", type=str,
        default="invar_rl/results/whole_stack_rl",
    )
    p.add_argument(
        "--panel_kind", type=str, default="lattice_native",
        choices=["biotech", "lattice_native"],
    )
    p.add_argument("--panel_end", type=str, default="2025-12-31")
    p.add_argument("--two_regime_val", action="store_true", default=True)
    p.add_argument("--total-timesteps", type=int, default=20000)
    p.add_argument("--universe-k", type=int, default=50)
    p.add_argument(
        "--feature-set", type=str, default="lite",
        choices=["lite", "rich"],
        help="lite=6 features (simplified); rich=15 features incl RSI/MACD/Bollinger",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir_root) / args.baseline
    run_one_cell(
        baseline=args.baseline,
        fold=args.fold,
        seed=args.seed,
        output_dir=out_dir,
        panel_kind=args.panel_kind,
        panel_end=args.panel_end,
        two_regime_val=args.two_regime_val,
        total_timesteps=args.total_timesteps,
        universe_k=args.universe_k,
        feature_set=args.feature_set,
    )


if __name__ == "__main__":
    main()
