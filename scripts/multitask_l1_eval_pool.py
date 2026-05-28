"""Aggregate Option B SP500 25-cell L2L3 outputs into pool / per-fold
annualised Sharpe and compare vs the canonical headline.

Reads ``outputs/sp500/stage3_rl_ablation/multitask_l1/foldF_seedS.json``
files written by ``multitask_l1_sp500_l2l3_25cell.sbatch`` and emits:

  * pool = mean over 25 cells of (mean_return * sqrt(252) / volatility)
  * per-fold pool (mean over 5 seeds)
  * canonical reference pool (read from the canonical_equal_l2_tape dir)
  * delta vs canonical

The pool definition matches the canonical headline reading at
``outputs/sp500/stage3_rl_ablation/canonical_equal_l2_tape/`` (annualised
per-cell Sharpe averaged over (fold, seed) cells = +0.9417 canonical).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


CANONICAL_DIR = (
    "outputs/sp500/stage3_rl_ablation/canonical_equal_l2_tape"
)
MULTITASK_DIR = (
    "outputs/sp500/stage3_rl_ablation/multitask_l1"
)


def _per_cell_sharpe(d: dict) -> Tuple[int, int, float, float, float]:
    """Return (fold, seed, mean_return, volatility, annualised_sharpe)."""
    sac = d["methods"]["sac"]
    mr = float(sac["mean_return"])
    vol = float(sac["volatility"])
    if vol <= 0.0:
        return int(d["fold"]), int(d["seed"]), mr, vol, float("nan")
    sharpe_ann = mr * np.sqrt(252.0) / vol
    return int(d["fold"]), int(d["seed"]), mr, vol, float(sharpe_ann)


def _aggregate(out_dir: Path) -> Tuple[float, Dict[int, float], List[dict]]:
    files = sorted(glob.glob(str(out_dir / "fold*_seed*.json")))
    if not files:
        return float("nan"), {}, []
    rows = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        fold, seed, mr, vol, sh = _per_cell_sharpe(d)
        rows.append({
            "fold": fold, "seed": seed, "mean_return": mr,
            "volatility": vol, "annualised_sharpe": sh,
            "path": f,
        })
    pool = float(np.nanmean([r["annualised_sharpe"] for r in rows]))
    by_fold: Dict[int, List[float]] = {}
    for r in rows:
        by_fold.setdefault(r["fold"], []).append(r["annualised_sharpe"])
    per_fold = {
        f: float(np.nanmean(v)) for f, v in sorted(by_fold.items())
    }
    return pool, per_fold, rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--canonical-dir", type=str, default=CANONICAL_DIR)
    p.add_argument("--multitask-dir", type=str, default=MULTITASK_DIR)
    args = p.parse_args()

    canon_pool, canon_per_fold, canon_rows = _aggregate(Path(args.canonical_dir))
    mt_pool, mt_per_fold, mt_rows = _aggregate(Path(args.multitask_dir))

    print("[INFO] canonical pool = %+.4f over %d cells" %
          (canon_pool, len(canon_rows)))
    for f in sorted(canon_per_fold):
        print(f"[INFO]   canonical fold {f} = {canon_per_fold[f]:+.4f}")
    print("[INFO] multitask pool = %+.4f over %d cells" %
          (mt_pool, len(mt_rows)))
    for f in sorted(mt_per_fold):
        print(f"[INFO]   multitask fold {f} = {mt_per_fold[f]:+.4f}")
    if not np.isnan(mt_pool) and not np.isnan(canon_pool):
        delta = mt_pool - canon_pool
        print(f"[INFO] delta multitask vs canonical = {delta:+.4f}")
        print(f"[INFO] stop-gate (>= +0.85 pool): "
              f"{'PASS' if mt_pool >= 0.85 else 'FAIL'}")
        f2 = mt_per_fold.get(2, float("nan"))
        print(f"[INFO] stop-gate (F2 >= -0.30): "
              f"{'PASS' if (not np.isnan(f2) and f2 >= -0.30) else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
