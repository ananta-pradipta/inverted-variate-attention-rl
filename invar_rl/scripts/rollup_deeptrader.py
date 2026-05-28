"""Roll up DeepTrader sweep results into the format used by Panel A.

Aggregates per-seed per-fold Sharpe ratios from the JSON cells under
``invar_rl/results/deeptrader/universal/`` and the per-seed DJIA
credibility cells under ``invar_rl/results/deeptrader/djia/``. Emits:

- Per-fold Sharpe (mean +/- std across seeds).
- Pooled-over-folds Sharpe (mean of all per-fold-per-seed values).
- DJIA credibility Sharpe (mean +/- std across seeds).

Usage::

    PYTHONPATH=$PWD python3 -m invar_rl.scripts.rollup_deeptrader

The rollup is printed to stdout; the paper prose is updated by hand.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


def _load(path: Path) -> dict:
    """Load a single JSON cell file.

    Args:
        path: Path to a ``foldN_seedM.json`` or ``djia_seedM.json`` file.

    Returns:
        Parsed JSON dict.
    """
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    """Walk the result tree and print the rollup."""
    root = Path("invar_rl/results/deeptrader")
    uni_dir = root / "universal"
    djia_dir = root / "djia"

    print("=" * 64)
    print("DeepTrader (rewritten 2026-05-22) sweep rollup")
    print("=" * 64)

    # Universal: per-fold-per-seed Sharpe table.
    by_fold: Dict[int, List[float]] = {}
    by_seed: Dict[int, List[float]] = {}
    all_sharpe: List[float] = []
    for p in sorted(uni_dir.glob("fold*_seed*.json")):
        d = _load(p)
        fold = int(d["fold"])
        seed = int(d["seed"])
        s = float(d["perf"]["sharpe"])
        by_fold.setdefault(fold, []).append(s)
        by_seed.setdefault(seed, []).append(s)
        all_sharpe.append(s)

    print()
    print("Universal 5-fold Sharpe (per fold, mean +/- std over seeds):")
    print(f"{'fold':>6} {'n':>4} {'mean':>10} {'std':>10}")
    for fold in sorted(by_fold):
        vals = by_fold[fold]
        m = mean(vals)
        s = stdev(vals) if len(vals) > 1 else 0.0
        print(f"{fold:>6} {len(vals):>4} {m:+10.4f} {s:>10.4f}")
    if all_sharpe:
        print(
            f"\npooled (over all fold*seed cells, N={len(all_sharpe)}):"
        )
        print(f"  mean={mean(all_sharpe):+.4f}")
        print(
            f"  std ={stdev(all_sharpe):.4f} "
            f"(over all {len(all_sharpe)} cells)"
        )

    # Universal per-seed pooled (this is what shows the bimodal collapse
    # pattern if any).
    print()
    print("Universal per-seed pooled Sharpe (mean over 5 folds):")
    print(f"{'seed':>6} {'n_folds':>8} {'pooled':>10}")
    for seed in sorted(by_seed):
        vals = by_seed[seed]
        print(f"{seed:>6} {len(vals):>8} {mean(vals):+10.4f}")

    # DJIA credibility.
    djia_sharpe: List[float] = []
    djia_eq: List[float] = []
    for p in sorted(djia_dir.glob("djia_seed*.json")):
        d = _load(p)
        djia_sharpe.append(float(d["perf"]["sharpe"]))
        djia_eq.append(float(d["perf"]["final_equity"]))
    print()
    print("DJIA-30 credibility (per seed):")
    print(f"  n        = {len(djia_sharpe)}")
    if djia_sharpe:
        print(f"  sharpes  = {[f'{s:+.3f}' for s in djia_sharpe]}")
        print(f"  mean     = {mean(djia_sharpe):+.4f}")
        if len(djia_sharpe) > 1:
            print(
                f"  std      = {stdev(djia_sharpe):.4f}"
            )
        print(
            f"  final eq = "
            f"{[f'{e:.3f}' for e in djia_eq]}"
        )


if __name__ == "__main__":
    main()
