"""Roll up the InVAR-RL-UR Phase 2 SP500 25-cell sweep.

Reads the per-cell summary jsons in
``outputs/sp500/layer2_ur/phase2_learned_gate/summary/foldF_seedS.json``
and prints:

  - per-cell Sharpe matrix (5 folds x 5 seeds)
  - per-fold mean +/- sd
  - pooled mean Sharpe across all 25 cells
  - comparison vs canonical SAC pooled +0.945 and per-fold table from
    ``reports/_rollup_fair_k25.json``
  - mechanism diagnostics (gate, gate_std, q_std, delta_abs, lcb) per
    fold and pooled across the 25 cells

Usage:
    python -m invar_rl.scripts.rollup_ur_sp500_phase2 \
        --out-root outputs/sp500/layer2_ur/phase2_learned_gate \
        --canonical reports/_rollup_fair_k25.json \
        --json-out reports/ur/_phase2_rollup.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple


_FOLDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)

_DIAG_KEYS: Tuple[str, ...] = (
    "gate", "gate_std", "q_std", "delta_abs", "lcb",
)


def _load_cell(summary_dir: Path, fold: int, seed: int) -> Optional[dict]:
    p = summary_dir / f"fold{fold}_seed{seed}.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def _safe_sd(xs: List[float]) -> float:
    return float(pstdev(xs)) if len(xs) > 1 else 0.0


def _canonical_per_fold(canon_path: Path) -> Tuple[Dict[int, float], float]:
    with open(canon_path) as fh:
        d = json.load(fh)
    blk = d["sp500"]["ls_canon_k50"]
    per_fold = {int(k): float(v) for k, v in blk["per_fold"].items()}
    pooled = float(blk["mean"])
    return per_fold, pooled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root", type=Path,
        default=Path("outputs/sp500/layer2_ur/phase2_learned_gate"),
    )
    ap.add_argument(
        "--canonical", type=Path,
        default=Path("reports/_rollup_fair_k25.json"),
    )
    ap.add_argument(
        "--json-out", type=Path,
        default=Path("reports/ur/_phase2_rollup.json"),
    )
    args = ap.parse_args()

    summary_dir = args.out_root / "summary"
    if not summary_dir.exists():
        raise SystemExit(f"summary dir missing: {summary_dir}")

    matrix: Dict[Tuple[int, int], Optional[float]] = {}
    diag_matrix: Dict[Tuple[int, int], Dict[str, Optional[float]]] = {}
    missing: List[Tuple[int, int]] = []
    for f in _FOLDS:
        for s in _SEEDS:
            payload = _load_cell(summary_dir, f, s)
            if payload is None:
                matrix[(f, s)] = None
                diag_matrix[(f, s)] = {k: None for k in _DIAG_KEYS}
                missing.append((f, s))
                continue
            matrix[(f, s)] = float(payload["test_pooled_sharpe"])
            ts = payload.get("ur_train_stats") or {}
            diag_matrix[(f, s)] = {k: float(ts.get(k)) if ts.get(k) is not None else None for k in _DIAG_KEYS}

    canon_per_fold, canon_pooled = _canonical_per_fold(args.canonical)

    print("# InVAR-RL-UR Phase 2 SP500 25-cell rollup")
    print()
    print(f"Source: {summary_dir}")
    print(f"Canonical reference: {args.canonical}")
    print()

    if missing:
        print(f"MISSING {len(missing)} cell(s): {missing}")
        print()

    print("Per-cell test pooled Sharpe (UR / canonical / delta):")
    print()
    header = "fold | " + " | ".join(f"seed {s}" for s in _SEEDS) + " | UR mean +/- sd | canonical | delta"
    print(header)
    print("-" * len(header))

    pooled_cells: List[float] = []
    fold_means: Dict[int, float] = {}
    fold_sds: Dict[int, float] = {}
    fold_deltas: Dict[int, float] = {}
    for f in _FOLDS:
        cells = [matrix[(f, s)] for s in _SEEDS]
        cells_present = [c for c in cells if c is not None]
        pooled_cells.extend(cells_present)
        m = mean(cells_present) if cells_present else float("nan")
        sd = _safe_sd(cells_present)
        fold_means[f] = m
        fold_sds[f] = sd
        canon = canon_per_fold[f]
        fold_deltas[f] = m - canon
        row_cells = [
            f"{c:+.3f}" if c is not None else "MISS" for c in cells
        ]
        print(
            f"F{f}   | " + " | ".join(row_cells)
            + f" | {m:+.3f} +/- {sd:.3f} | {canon:+.3f} | {m - canon:+.3f}"
        )

    pooled_mean = mean(pooled_cells) if pooled_cells else float("nan")
    pooled_sd = _safe_sd(pooled_cells)
    pooled_sem = pooled_sd / math.sqrt(len(pooled_cells)) if pooled_cells else 0.0
    delta_pooled = pooled_mean - canon_pooled

    print()
    print(
        f"Pooled UR ({len(pooled_cells)} cells): mean={pooled_mean:+.4f}, "
        f"sd={pooled_sd:.4f}, sem={pooled_sem:.4f}"
    )
    print(f"Canonical SAC pooled (25 cells): {canon_pooled:+.4f}")
    print(f"Delta (UR - canonical) pooled: {delta_pooled:+.4f}")
    if pooled_sem > 0:
        print(f"  in sem units: {delta_pooled / pooled_sem:+.2f}")
    print()

    print("Per-fold mean Sharpe (UR vs canonical):")
    for f in _FOLDS:
        print(
            f"  F{f}: UR {fold_means[f]:+.3f}  canonical {canon_per_fold[f]:+.3f}  "
            f"delta {fold_deltas[f]:+.3f}"
        )
    print()

    print("Mechanism diagnostics (mean across present cells):")
    print(f"{'fold':<6} " + " ".join(f"{k:>10}" for k in _DIAG_KEYS))
    pooled_diag: Dict[str, List[float]] = {k: [] for k in _DIAG_KEYS}
    fold_diag: Dict[int, Dict[str, float]] = {}
    for f in _FOLDS:
        row = []
        fold_diag[f] = {}
        for k in _DIAG_KEYS:
            vals = [
                diag_matrix[(f, s)][k]
                for s in _SEEDS
                if diag_matrix[(f, s)][k] is not None
            ]
            if vals:
                m = mean(vals)
                pooled_diag[k].extend(vals)
                fold_diag[f][k] = float(m)
                row.append(f"{m:>+10.4f}")
            else:
                fold_diag[f][k] = float("nan")
                row.append(f"{'MISS':>10}")
        print(f"F{f}     " + " ".join(row))

    print(f"{'pool':<6} " + " ".join(
        (f"{mean(pooled_diag[k]):>+10.4f}" if pooled_diag[k] else f"{'MISS':>10}")
        for k in _DIAG_KEYS
    ))

    # JSON dump
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "source_dir": str(summary_dir),
        "canonical_path": str(args.canonical),
        "n_present_cells": len(pooled_cells),
        "n_missing_cells": len(missing),
        "missing": [list(t) for t in missing],
        "per_cell_sharpe": {
            f"fold{f}_seed{s}": matrix[(f, s)]
            for f in _FOLDS for s in _SEEDS
        },
        "per_fold_mean": {str(f): fold_means[f] for f in _FOLDS},
        "per_fold_sd": {str(f): fold_sds[f] for f in _FOLDS},
        "per_fold_canonical": {str(f): canon_per_fold[f] for f in _FOLDS},
        "per_fold_delta": {str(f): fold_deltas[f] for f in _FOLDS},
        "pooled_mean": pooled_mean,
        "pooled_sd": pooled_sd,
        "pooled_sem": pooled_sem,
        "canonical_pooled": canon_pooled,
        "pooled_delta": delta_pooled,
        "diagnostics_per_fold": fold_diag,
        "diagnostics_pooled": {
            k: (float(mean(pooled_diag[k])) if pooled_diag[k] else None)
            for k in _DIAG_KEYS
        },
    }
    with open(args.json_out, "w") as fh:
        json.dump(out_payload, fh, indent=2)
    print()
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
