"""Robust-InVAR-RL Phase 2 rollup: SP500 25-cell + bootstrap p-values.

Reads:
- Phase 2 proposed cells: ``outputs/sp500/layer3_robust_phase2_25cell/
  foldF_seedS.json`` (summary) and
  ``daily_tape/sac/foldF_seedS.parquet`` (daily tape).
- Phase 1 proposed cells: ``outputs/sp500/stage3_rl_ablation/
  robust_phase1/foldF_seedS.json`` (summary) and
  ``daily_tape/sac/foldF_seedS.parquet`` (daily tape).
- Canonical baseline cells: ``invar_rl/results/stage3_rl_ablation/
  equal_l2/foldF_seedS.json`` (summary). Daily-tape canonical lives at
  ``outputs/sp500/stage3_rl_ablation/canonical_equal_l2_tape_25cell/
  daily_tape/sac/foldF_seedS.parquet`` if the Phase 1 canonical tape
  sbatch has run.

Produces a per-fold table for Phase 2 vs Phase 1 and Phase 2 vs
canonical, pool deltas, and the Phase 2 stop-gate verdict per the
source design doc + the user's Phase 2 instructions.

Usage:
    python scripts/rollup_robust_phase2.py \\
        --phase2-dir outputs/sp500/layer3_robust_phase2_25cell \\
        --phase1-dir outputs/sp500/stage3_rl_ablation/robust_phase1 \\
        --baseline-dir invar_rl/results/stage3_rl_ablation/equal_l2 \\
        --phase2-tape-dir outputs/sp500/layer3_robust_phase2_25cell/daily_tape/sac \\
        --phase1-tape-dir outputs/sp500/stage3_rl_ablation/robust_phase1/daily_tape/sac \\
        --baseline-tape-dir outputs/sp500/stage3_rl_ablation/canonical_equal_l2_tape_25cell/daily_tape/sac \\
        --out reports/robust_invar_rl/phase_2_residual_sac_25cell.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Canonical Phase 0 references (locked 2026-05-26).
CANONICAL_POOLED_SHARPE_K50 = 0.945
CANONICAL_F1 = 0.855
CANONICAL_F2 = -0.229
CANONICAL_F3 = 0.862
CANONICAL_F4 = 1.155
CANONICAL_F5 = 2.066
PHASE1_POOLED = 0.665
PHASE1_F2 = 0.012

TRADING_DAYS = 252

# Phase 2 stop-gate thresholds (per source doc + user instructions).
PHASE2_F4F5_RECOVERY_FLOOR_DELTA = -0.30   # F4+F5 sum >= canonical sum - 0.30
PHASE2_F2_FLOOR = -0.10                    # F2 >= -0.10
PHASE2_POOL_FLOOR_DELTA = -0.05            # pool within -0.05 of canonical
PHASE2_POOL_HARD_FAIL_DELTA = -0.10        # > -0.10 below canonical => hard fail
PHASE2_BOOTSTRAP_REQUIRED_FOLDS_VS_PHASE1 = 1


def _sharpe(ret: np.ndarray) -> float:
    if ret.size < 2:
        return 0.0
    sd = float(ret.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(ret.mean() / sd * math.sqrt(TRADING_DAYS))


def _cell_sharpe_from_json(path: str) -> float:
    with open(path) as fh:
        payload = json.load(fh)
    if "methods" not in payload or "sac" not in payload["methods"]:
        return 0.0
    sac = payload["methods"]["sac"]
    mean = float(sac.get("mean_return", 0.0))
    vol = float(sac.get("volatility", 0.0))
    if vol <= 1e-12:
        return 0.0
    return mean / vol * math.sqrt(TRADING_DAYS)


def _load_daily(parquet_path: str) -> np.ndarray:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    return df["strategy_return"].to_numpy(dtype=np.float64)


def _paired_bootstrap_p(
    a: np.ndarray, b: np.ndarray,
    block_length: int = 5, n_reps: int = 1000, seed: int = 42,
) -> Dict[str, float]:
    """Paired stationary-bootstrap p-value for H0: Sharpe(a) <= Sharpe(b)."""
    n = int(min(a.shape[0], b.shape[0]))
    if n < 5:
        return {
            "sharpe_a": float(_sharpe(a)),
            "sharpe_b": float(_sharpe(b)),
            "delta": float(_sharpe(a) - _sharpe(b)),
            "p": float("nan"),
            "n_days": int(n),
        }
    a = a[:n]
    b = b[:n]
    point_a = _sharpe(a)
    point_b = _sharpe(b)
    delta_point = point_a - point_b
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_reps, dtype=np.float64)
    p_cont = 1.0 - 1.0 / float(max(1, block_length))
    for r in range(n_reps):
        idxs = np.empty(n, dtype=np.int64)
        cur = int(rng.integers(0, n))
        for t in range(n):
            idxs[t] = cur
            if rng.random() < p_cont:
                cur = (cur + 1) % n
            else:
                cur = int(rng.integers(0, n))
        deltas[r] = _sharpe(a[idxs]) - _sharpe(b[idxs])
    if delta_point >= 0:
        p = float((deltas <= 0).mean())
    else:
        p = float((deltas >= 0).mean())
    return {
        "sharpe_a": float(point_a),
        "sharpe_b": float(point_b),
        "delta": float(delta_point),
        "p": float(p),
        "n_days": int(n),
    }


def _collect_cells(base_glob: str, sharpe_fn) -> Dict[Tuple[int, int], Tuple[float, str]]:
    out: Dict[Tuple[int, int], Tuple[float, str]] = {}
    for fp in sorted(glob.glob(base_glob)):
        name = Path(fp).stem
        try:
            f = int(name.split("fold")[1].split("_")[0])
            s = int(name.split("seed")[1])
        except Exception:
            continue
        out[(f, s)] = (float(sharpe_fn(fp)), fp)
    return out


def _per_fold_means(
    cells: Dict[Tuple[int, int], Tuple[float, str]],
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for f in [1, 2, 3, 4, 5]:
        vals = [v for ((ff, _), (v, _)) in cells.items() if ff == f]
        if vals:
            out[f] = float(np.mean(vals))
    return out


def _fold_bootstrap(
    phase2_tape_dir: str, ref_tape_dir: str, seeds_per_fold: int = 5,
    block_length: int = 5, n_reps: int = 1000,
) -> Dict[int, float]:
    """Per-fold min p-value across seeds, comparing phase2 vs ref."""
    out: Dict[int, float] = {}
    for f in [1, 2, 3, 4, 5]:
        p_vals: List[float] = []
        for seed in [42, 43, 44, 45, 46][:seeds_per_fold]:
            a_path = Path(phase2_tape_dir) / f"fold{f}_seed{seed}.parquet"
            b_path = Path(ref_tape_dir) / f"fold{f}_seed{seed}.parquet"
            if not a_path.exists() or not b_path.exists():
                continue
            a = _load_daily(str(a_path))
            b = _load_daily(str(b_path))
            res = _paired_bootstrap_p(
                a, b, block_length=block_length, n_reps=n_reps,
                seed=42 + seed,
            )
            p_vals.append(float(res["p"]))
        out[f] = float(min(p_vals)) if p_vals else float("nan")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase2-dir", type=str,
        default="outputs/sp500/layer3_robust_phase2_25cell",
    )
    p.add_argument(
        "--phase1-dir", type=str,
        default="outputs/sp500/stage3_rl_ablation/robust_phase1",
    )
    p.add_argument(
        "--baseline-dir", type=str,
        default="invar_rl/results/stage3_rl_ablation/equal_l2",
    )
    p.add_argument(
        "--phase2-tape-dir", type=str,
        default="outputs/sp500/layer3_robust_phase2_25cell/daily_tape/sac",
    )
    p.add_argument(
        "--phase1-tape-dir", type=str,
        default="outputs/sp500/stage3_rl_ablation/robust_phase1/daily_tape/sac",
    )
    p.add_argument(
        "--baseline-tape-dir", type=str,
        default="outputs/sp500/stage3_rl_ablation/canonical_equal_l2_tape_25cell/daily_tape/sac",
        help="Optional; canonical baseline daily tape if available.",
    )
    p.add_argument(
        "--out", type=str,
        default="reports/robust_invar_rl/phase_2_residual_sac_25cell.md",
    )
    p.add_argument("--block-length", type=int, default=5)
    p.add_argument("--n-reps", type=int, default=1000)
    args = p.parse_args()

    p2 = _collect_cells(
        str(Path(args.phase2_dir) / "fold*_seed*.json"),
        _cell_sharpe_from_json,
    )
    p1 = _collect_cells(
        str(Path(args.phase1_dir) / "fold*_seed*.json"),
        _cell_sharpe_from_json,
    )
    bl = _collect_cells(
        str(Path(args.baseline_dir) / "fold*_seed*.json"),
        _cell_sharpe_from_json,
    )

    p2_per_fold = _per_fold_means(p2)
    p1_per_fold = _per_fold_means(p1)
    bl_per_fold = _per_fold_means(bl)

    pool_p2 = float(np.mean(list(p2_per_fold.values()))) if p2_per_fold else float("nan")
    pool_p1 = float(np.mean(list(p1_per_fold.values()))) if p1_per_fold else float("nan")
    pool_bl = float(np.mean(list(bl_per_fold.values()))) if bl_per_fold else float("nan")

    p_vs_p1 = _fold_bootstrap(
        args.phase2_tape_dir, args.phase1_tape_dir,
        block_length=args.block_length, n_reps=args.n_reps,
    )
    p_vs_bl = (
        _fold_bootstrap(
            args.phase2_tape_dir, args.baseline_tape_dir,
            block_length=args.block_length, n_reps=args.n_reps,
        ) if Path(args.baseline_tape_dir).exists() else {}
    )

    lines: List[str] = []
    lines.append("# Robust-InVAR-RL Phase 2: 25-cell SP500 rollup")
    lines.append("")
    lines.append(
        f"Canonical equal_l2 SAC K=50: pool +{CANONICAL_POOLED_SHARPE_K50:.3f}, "
        f"F2 {CANONICAL_F2:+.3f}, F4 {CANONICAL_F4:+.3f}, F5 {CANONICAL_F5:+.3f}"
    )
    lines.append(
        f"Phase 1 (group-DRO ranker + canonical SAC): pool +{PHASE1_POOLED:.3f}, "
        f"F2 {PHASE1_F2:+.3f}"
    )
    lines.append("")
    lines.append("## Per-fold table")
    lines.append("")
    lines.append(
        "| Fold | Canonical | Phase 1 | Phase 2 | "
        "Delta P2-P1 | Delta P2-Canon | "
        "Boot p (P2 vs P1) | Boot p (P2 vs Canon) |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for f in [1, 2, 3, 4, 5]:
        canon = bl_per_fold.get(f, float("nan"))
        ph1 = p1_per_fold.get(f, float("nan"))
        ph2 = p2_per_fold.get(f, float("nan"))
        d_p1 = (ph2 - ph1) if not (math.isnan(ph2) or math.isnan(ph1)) else float("nan")
        d_bl = (ph2 - canon) if not (math.isnan(ph2) or math.isnan(canon)) else float("nan")
        pp1 = p_vs_p1.get(f, float("nan"))
        ppb = p_vs_bl.get(f, float("nan"))
        lines.append(
            f"| F{f} | {canon:+.4f} | {ph1:+.4f} | {ph2:+.4f} | "
            f"{d_p1:+.4f} | {d_bl:+.4f} | "
            f"{pp1:.3f} | {ppb:.3f} |"
        )

    pool_delta_p1 = pool_p2 - pool_p1
    pool_delta_bl = pool_p2 - pool_bl
    f2_p2 = p2_per_fold.get(2, float("nan"))
    f4_p2 = p2_per_fold.get(4, float("nan"))
    f5_p2 = p2_per_fold.get(5, float("nan"))
    f4f5_sum_canon = CANONICAL_F4 + CANONICAL_F5
    f4f5_sum_p2 = (f4_p2 + f5_p2) if not (math.isnan(f4_p2) or math.isnan(f5_p2)) else float("nan")
    f4f5_delta = f4f5_sum_p2 - f4f5_sum_canon if not math.isnan(f4f5_sum_p2) else float("nan")

    lines.append("")
    lines.append(
        f"**Pool**: Phase 2={pool_p2:+.4f}, Phase 1={pool_p1:+.4f}, "
        f"Canonical={pool_bl:+.4f}. "
        f"Delta P2-P1={pool_delta_p1:+.4f}, Delta P2-Canon={pool_delta_bl:+.4f}."
    )
    lines.append(
        f"**F2**: Phase 2={f2_p2:+.4f} (canonical {CANONICAL_F2:+.3f}, Phase 1 {PHASE1_F2:+.3f})."
    )
    lines.append(
        f"**F4+F5 sum**: Phase 2={f4f5_sum_p2:+.4f} "
        f"(canonical {f4f5_sum_canon:+.4f}); delta={f4f5_delta:+.4f}."
    )
    lines.append("")
    lines.append("## Phase 2 stop-gate verdict")
    lines.append("")
    gates: List[Tuple[str, bool, str]] = []
    f4f5_pass = (
        not math.isnan(f4f5_sum_p2)
        and f4f5_sum_p2 >= f4f5_sum_canon + PHASE2_F4F5_RECOVERY_FLOOR_DELTA
    )
    gates.append((
        "F4+F5 recover (sum >= canonical - 0.30)",
        f4f5_pass,
        f"F4+F5={f4f5_sum_p2:+.4f} vs floor {f4f5_sum_canon + PHASE2_F4F5_RECOVERY_FLOOR_DELTA:+.4f}",
    ))
    f2_pass = (not math.isnan(f2_p2)) and (f2_p2 >= PHASE2_F2_FLOOR)
    gates.append((
        f"F2 preserved (>= {PHASE2_F2_FLOOR:+.2f})",
        f2_pass,
        f"F2={f2_p2:+.4f}",
    ))
    pool_pass = (not math.isnan(pool_delta_bl)) and pool_delta_bl >= PHASE2_POOL_FLOOR_DELTA
    gates.append((
        "Pool within -0.05 of canonical",
        pool_pass,
        f"delta vs canonical={pool_delta_bl:+.4f}",
    ))
    n_p1_under = sum(
        1 for v in p_vs_p1.values()
        if not math.isnan(v) and v < 0.10
    )
    boot_pass = n_p1_under >= PHASE2_BOOTSTRAP_REQUIRED_FOLDS_VS_PHASE1
    gates.append((
        "Bootstrap p vs Phase 1 < 0.10 on >= 1 fold",
        boot_pass,
        f"{n_p1_under}/5 folds with p<0.10 vs Phase 1",
    ))
    hard_fail = (not math.isnan(pool_delta_bl)) and pool_delta_bl < PHASE2_POOL_HARD_FAIL_DELTA
    if hard_fail:
        gates.append((
            "HARD FAIL: pool dropped > 0.10 below canonical",
            False,
            f"pool delta vs canonical={pool_delta_bl:+.4f}",
        ))
    f2_hard_fail = (not math.isnan(f2_p2)) and f2_p2 <= CANONICAL_F2 + 0.01
    if f2_hard_fail:
        gates.append((
            "HARD FAIL: F2 regressed to canonical",
            False,
            f"F2={f2_p2:+.4f} vs canonical {CANONICAL_F2:+.3f}",
        ))

    for name, ok, note in gates:
        status = "PASS" if ok else "FAIL"
        lines.append(f"- [{status}] {name}: {note}")
    overall = all(ok for _, ok, _ in gates)
    lines.append("")
    lines.append(f"**Overall verdict**: {'PASS' if overall else 'FAIL'}")
    lines.append("")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[INFO] wrote {out_path}")
    for ln in lines:
        print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
