"""Roll up SP500 + NDX-100 InVAR-RL Phase 5 SAC K=25 fair-K runs.

Computes per-cell mean Sharpe + per-fold breakdown for both universes
at K=25 (equal-weight wrapper) and compares to the canonical numbers:

    SP500 L/S K=50:  +0.945 (equal_l2 ablation, SAC entry)
    SP500 L/O K=50:  +0.554 (stage3_rl_long_only, QP)
    NDX-100 L/S K=20: +1.194 (canonical SAC, QP)
    NDX-100 L/O K=20: +0.657 (canonical SAC, QP)

Per-cell Sharpe is annualised (* sqrt(252)) and pooled by per-cell
mean across the 25 (fold x seed) cells. This matches the convention
used in reports/invar-rl-experiment-result-all.md.

Usage::

    python scripts/rollup_fair_k25.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_TRADING_DAYS = 252


def _sp500_cell_sharpe(path: str) -> float:
    """Sharpe for a SP500 stage3_rl_ablation JSON (SAC entry)."""
    with open(path) as fh:
        payload = json.load(fh)
    sac = payload["methods"]["sac"]
    mean = float(sac["mean_return"])
    vol = float(sac["volatility"])
    if vol <= 1e-12:
        return 0.0
    return mean / vol * math.sqrt(_TRADING_DAYS)


def _ndx_cell_sharpe(path: str) -> float:
    """Sharpe for an NDX-100 layer3_sac summary JSON."""
    with open(path) as fh:
        payload = json.load(fh)
    return float(payload.get("test_pooled_sharpe", 0.0))


def _roll(
    glob_pattern: str,
    sharpe_fn,
    label: str,
) -> Dict[str, object]:
    """Roll up per-cell Sharpes; return mean, sd, per-fold means, count."""
    cells: List[Tuple[int, int, float]] = []  # (fold, seed, sharpe)
    for fp in sorted(glob.glob(glob_pattern)):
        # Filenames are foldF_seedS.json. Parse F, S.
        name = Path(fp).stem
        try:
            fold = int(name.split("fold")[1].split("_")[0])
            seed = int(name.split("seed")[1])
        except Exception:
            continue
        cells.append((fold, seed, sharpe_fn(fp)))
    if not cells:
        return {"label": label, "n": 0}
    sharpes = np.asarray([s for _, _, s in cells], dtype=np.float64)
    fold_means = {}
    for f in (1, 2, 3, 4, 5):
        fold_vals = [s for (ff, _, s) in cells if ff == f]
        fold_means[f] = float(np.mean(fold_vals)) if fold_vals else float("nan")
    return {
        "label": label,
        "n": int(sharpes.size),
        "mean": float(sharpes.mean()),
        "sd": float(sharpes.std(ddof=1)) if sharpes.size > 1 else 0.0,
        "per_fold": fold_means,
        "n_per_fold": {
            f: sum(1 for (ff, _, _) in cells if ff == f) for f in (1, 2, 3, 4, 5)
        },
    }


def _fmt(roll: Dict[str, object], baseline_mean: float | None = None) -> str:
    if roll["n"] == 0:
        return f"{roll['label']:50s}  n=0  (no cells)"
    pf = roll["per_fold"]
    line = (
        f"{roll['label']:50s}  "
        f"n={roll['n']:>2d}  "
        f"mean={roll['mean']:+.4f}  "
        f"sd={roll['sd']:.3f}  "
        f"F1={pf[1]:+.3f}  F2={pf[2]:+.3f}  F3={pf[3]:+.3f}  "
        f"F4={pf[4]:+.3f}  F5={pf[5]:+.3f}"
    )
    if baseline_mean is not None:
        delta = roll["mean"] - baseline_mean
        line += f"  delta_vs_baseline={delta:+.4f}"
    return line


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    print("=" * 110)
    print("InVAR-RL Phase 5 SAC fair-K (K=25) rollup vs canonical K=50/K=20")
    print("=" * 110)

    # Canonical reference numbers
    sp_ls_canon = _roll(
        str(repo / "invar_rl/results/stage3_rl_ablation/equal_l2/fold*_seed*.json"),
        _sp500_cell_sharpe,
        "SP500 L/S K=50 (canonical equal_l2 SAC)",
    )
    sp_lo_canon = _roll(
        str(repo / "invar_rl/results/stage3_rl_long_only/fold*_seed*.json"),
        _sp500_cell_sharpe,
        "SP500 L/O K=50 (canonical stage3_rl_long_only QP)",
    )
    ndx_ls_canon = _roll(
        str(repo / "outputs/nasdaq100/layer3/ls/summary/fold*_seed*.json"),
        _ndx_cell_sharpe,
        "NDX-100 L/S K=20 (canonical layer3 QP)",
    )
    ndx_lo_canon = _roll(
        str(repo / "outputs/nasdaq100/layer3/lo/summary/fold*_seed*.json"),
        _ndx_cell_sharpe,
        "NDX-100 L/O K=20 (canonical layer3 QP)",
    )

    print(_fmt(sp_ls_canon))
    print(_fmt(sp_lo_canon))
    print(_fmt(ndx_ls_canon))
    print(_fmt(ndx_lo_canon))
    print()

    # K=25 fair-K runs
    sp_ls_k25 = _roll(
        str(repo / "outputs/sp500/layer3_k25/ls/fold*_seed*.json"),
        _sp500_cell_sharpe,
        "SP500 L/S K=25 (fair-K equal_l2 SAC)",
    )
    sp_lo_k25 = _roll(
        str(repo / "outputs/sp500/layer3_k25/lo/fold*_seed*.json"),
        _sp500_cell_sharpe,
        "SP500 L/O K=25 (fair-K equal_l2 SAC, long-only)",
    )
    ndx_ls_k25 = _roll(
        str(repo / "outputs/nasdaq100/layer3_k25/ls/summary/fold*_seed*.json"),
        _ndx_cell_sharpe,
        "NDX-100 L/S K=25 (fair-K equal_topk SAC)",
    )
    ndx_lo_k25 = _roll(
        str(repo / "outputs/nasdaq100/layer3_k25/lo/summary/fold*_seed*.json"),
        _ndx_cell_sharpe,
        "NDX-100 L/O K=25 (fair-K equal_topk SAC)",
    )

    print(_fmt(sp_ls_k25, baseline_mean=sp_ls_canon.get("mean")))
    print(_fmt(sp_lo_k25, baseline_mean=sp_lo_canon.get("mean")))
    print(_fmt(ndx_ls_k25, baseline_mean=ndx_ls_canon.get("mean")))
    print(_fmt(ndx_lo_k25, baseline_mean=ndx_lo_canon.get("mean")))

    # JSON dump for downstream paper / report patching
    out = {
        "sp500": {
            "ls_canon_k50": sp_ls_canon,
            "lo_canon_k50": sp_lo_canon,
            "ls_fair_k25": sp_ls_k25,
            "lo_fair_k25": sp_lo_k25,
        },
        "nasdaq100": {
            "ls_canon_k20": ndx_ls_canon,
            "lo_canon_k20": ndx_lo_canon,
            "ls_fair_k25": ndx_ls_k25,
            "lo_fair_k25": ndx_lo_k25,
        },
    }
    out_path = repo / "reports/_rollup_fair_k25.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print()
    print(f"JSON dump written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
