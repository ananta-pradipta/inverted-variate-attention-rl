"""Roll up Option F (SAC curriculum) per-cell finetune results.

Reads the 25 per-(fold, seed) JSON files at
``outputs/sac_curriculum/sp500/per_cell_finetune/fold{F}_seed{S}.json``
and prints the per-fold means, the 25-cell pool, and the delta vs the
canonical SP500 SAC baseline (pool +0.945, F2 -0.229).

Usage::

    PYTHONPATH=$PWD python3 -m invar_rl.scripts.rollup_sac_curriculum
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional


_TRADING_DAYS = 252
_FOLDS = (1, 2, 3, 4, 5)
_SEEDS = (42, 43, 44, 45, 46)

_CANONICAL_POOLED = 0.9417
_CANONICAL_PER_FOLD: Dict[int, float] = {
    1: 0.8552,
    2: -0.2292,
    3: 0.8618,
    4: 1.1549,
    5: 2.0656,
}


def _per_cell_sharpe(payload: dict) -> Optional[float]:
    if "methods" not in payload or "sac" not in payload["methods"]:
        return None
    perf = payload["methods"]["sac"]
    mr = float(perf.get("mean_return", 0.0))
    vol = float(perf.get("volatility", 0.0))
    if vol <= 1e-12:
        return 0.0
    return mr / vol * math.sqrt(_TRADING_DAYS)


def _load_cells(root: Path) -> Dict[tuple, float]:
    cells: Dict[tuple, float] = {}
    for fold in _FOLDS:
        for seed in _SEEDS:
            p = root / f"fold{fold}_seed{seed}.json"
            if not p.exists():
                continue
            with open(p) as fh:
                payload = json.load(fh)
            sh = _per_cell_sharpe(payload)
            if sh is None:
                continue
            cells[(fold, seed)] = sh
    return cells


def _summarise(cells: Dict[tuple, float]) -> Dict[str, object]:
    per_fold: Dict[int, List[float]] = {f: [] for f in _FOLDS}
    pooled: List[float] = []
    for (fold, seed), sh in cells.items():
        per_fold[fold].append(sh)
        pooled.append(sh)
    fold_means = {
        f: (mean(v) if v else float("nan"))
        for f, v in per_fold.items()
    }
    pool_mean = mean(pooled) if pooled else float("nan")
    pool_sd = pstdev(pooled) if len(pooled) > 1 else 0.0
    return {
        "n_cells": len(cells),
        "pool_mean": pool_mean,
        "pool_sd": pool_sd,
        "per_fold_mean": fold_means,
        "per_fold_n": {f: len(v) for f, v in per_fold.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-root", type=str,
        default="outputs/sac_curriculum/sp500/per_cell_finetune",
    )
    args = p.parse_args()
    root = Path(args.results_root)
    cells = _load_cells(root)
    summary = _summarise(cells)
    print(f"[Option F rollup] results_root={root}")
    print(
        f"  n_cells={summary['n_cells']}/25  "
        f"pool_mean={summary['pool_mean']:+.4f}  "
        f"pool_sd={summary['pool_sd']:.4f}"
    )
    print(
        f"  vs canonical pooled {_CANONICAL_POOLED:+.3f}: "
        f"delta = {summary['pool_mean'] - _CANONICAL_POOLED:+.4f}"
    )
    print("  per-fold means (n):")
    for f in _FOLDS:
        v = summary["per_fold_mean"][f]
        n = summary["per_fold_n"][f]
        canon = _CANONICAL_PER_FOLD[f]
        delta = v - canon
        print(
            f"    F{f}: {v:+.4f} (n={n:>2})  "
            f"canonical {canon:+.3f}  delta {delta:+.4f}"
        )
    print("  per-cell:")
    for f in _FOLDS:
        for s in _SEEDS:
            sh = cells.get((f, s))
            tag = f"{sh:+.4f}" if sh is not None else "MISSING"
            print(f"    fold{f}_seed{s}: {tag}")


if __name__ == "__main__":
    main()
