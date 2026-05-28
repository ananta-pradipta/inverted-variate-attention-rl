"""Evaluate the non-learning baselines on the canonical InVAR-RL test folds.

For each (fold) cell, builds the lattice_native panel via the same data
bridge canonical InVAR uses, runs the non-learning strategies in
:mod:`invar_rl.baselines.non_learning`, and writes a JSON output with
the same metric schema as ``stage3_eval`` / ``stage3_rl_canonical`` so
the comparison table in the paper is paper-grade.

Output: ``invar_rl/results/non_learning_baselines/foldF.json`` with one
sub-dict per baseline, plus a top-level ``methods`` block matching the
RL output schema for direct table merge.

Usage::

    python -m invar_rl.training.non_learning_eval --fold 1
    python -m invar_rl.training.non_learning_eval --fold 1 --segment val
    # default segment is "test"

These baselines are deterministic given the panel; no seed loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.invar import InVARConfig

from invar_rl.baselines.non_learning import (
    buy_and_hold,
    equal_weight_long,
    momentum_long_short,
    reversal_long_short,
    volatility_targeted_market,
)
from invar_rl.data.lattice_bridge import build_lattice_bridge


def run_one_fold(
    fold: int,
    segment: str,
    output_dir: Path,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
) -> dict:
    cfg = InVARConfig(fold=fold, seed=42)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    bridge = build_lattice_bridge(cfg)
    if segment == "test":
        day_indices: Sequence[int] = list(bridge.test_idx)
    elif segment == "val":
        day_indices = list(bridge.val_idx)
    elif segment == "train":
        day_indices = list(bridge.train_idx)
    else:
        raise ValueError(f"unknown segment {segment!r}")
    print(
        f"[non-learning eval] fold={fold} segment={segment} "
        f"n_days={len(day_indices)}"
    )

    strategies = [
        ("buy_and_hold", buy_and_hold(bridge, day_indices)),
        ("equal_weight_long", equal_weight_long(bridge, day_indices)),
        ("momentum_jt_12_2", momentum_long_short(
            bridge, day_indices, lookback=252, skip=21,
        )),
        ("reversal_1m", reversal_long_short(
            bridge, day_indices, lookback=21,
        )),
        ("vol_targeted_market_10", volatility_targeted_market(
            bridge, day_indices, target_ann_vol=0.10,
        )),
    ]

    methods = {}
    for name, res in strategies:
        d = res.as_dict()
        methods[name] = d
        print(
            f"  {name:30s} sharpe={d['sharpe_annualised']:+.3f} "
            f"ann_ret={d['mean_return']*252:+.4f} "
            f"ann_vol={d['volatility']*(252**0.5):+.4f} "
            f"eq={d['final_equity']:.4f}"
        )

    payload = {
        "fold": fold,
        "segment": segment,
        "panel_kind": panel_kind,
        "two_regime_val": two_regime_val,
        "panel_end": panel_end,
        "n_days": len(day_indices),
        "methods": methods,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fold{fold}_{segment}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[non-learning eval] wrote {out_path}")
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Non-learning baselines on InVAR-RL folds (deterministic)."
    )
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument(
        "--segment", type=str, default="test",
        choices=["train", "val", "test"],
    )
    p.add_argument(
        "--output-dir", type=str,
        default="invar_rl/results/non_learning_baselines",
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
    run_one_fold(
        fold=args.fold,
        segment=args.segment,
        output_dir=Path(args.output_dir),
        panel_kind=args.panel_kind,
        panel_end=args.panel_end,
        two_regime_val=args.two_regime_val,
    )


if __name__ == "__main__":
    main()
