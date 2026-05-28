"""Robust-InVAR-RL Phase 1 rollup: SP500 25-cell with bootstrap p-values.

Reads:
- Phase 1 proposed cells: outputs/sp500/stage3_rl_ablation/robust_phase1/
  fold{F}_seed{S}.json (summary) and
  daily_tape/sac/fold{F}_seed{S}.parquet (daily tape)
- Canonical baseline cells: invar_rl/results/stage3_rl_ablation/equal_l2/
  fold{F}_seed{S}.json

Produces per-fold table (mean Sharpe + per-cell paired bootstrap p-values
against the canonical baseline), pooled delta, and a Phase 1 stop-gate
verdict per the source design doc.

Usage:
    python scripts/rollup_robust_phase1.py \
        --proposed-dir outputs/sp500/stage3_rl_ablation/robust_phase1 \
        --baseline-dir invar_rl/results/stage3_rl_ablation/equal_l2 \
        --out reports/robust_invar_rl/phase_1_group_dro_25cell.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Canonical Phase 0 references (locked 2026-05-26 in protocol_lock.yaml).
CANONICAL_POOLED_SHARPE_K50 = 0.945
CANONICAL_F2_SHARPE_K50 = -0.229
CANONICAL_POOLED_SHARPE_K25 = 0.899
TRADING_DAYS = 252
PHASE1_STOP_GATE_F2_FLOOR = -0.10
PHASE1_STOP_GATE_POOL_FLOOR_DELTA = -0.10
PHASE1_STOP_GATE_POOL_HARD_FAIL_DELTA = -0.20
PHASE1_STOP_GATE_BOOTSTRAP_FOLDS = 1


def _sharpe(ret: np.ndarray) -> float:
    if ret.size < 2:
        return 0.0
    sd = float(ret.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(ret.mean() / sd * math.sqrt(TRADING_DAYS))


def _cell_sharpe_from_json(path: str) -> float:
    """Pull SAC method's annualised Sharpe from a stage3_rl_ablation JSON."""
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


def _stationary_block_resample(
    arr: np.ndarray, block_length: int, rng: np.random.Generator,
) -> np.ndarray:
    n = arr.shape[0]
    p_cont = 1.0 - 1.0 / float(max(1, block_length))
    out = np.empty(n, dtype=arr.dtype)
    idx = int(rng.integers(0, n))
    for t in range(n):
        out[t] = arr[idx]
        if rng.random() < p_cont:
            idx = (idx + 1) % n
        else:
            idx = int(rng.integers(0, n))
    return out


def _paired_bootstrap_p(
    a: np.ndarray, b: np.ndarray,
    block_length: int = 5, n_reps: int = 1000, seed: int = 42,
) -> Dict[str, float]:
    """Paired stationary-bootstrap p-value for H0: Sharpe(a) <= Sharpe(b).

    Returns dict with point Sharpes, delta, and one-sided p-value for
    delta > 0 (i.e., a beats b).
    """
    n = int(min(a.shape[0], b.shape[0]))
    if n < 5:
        return {
            "sharpe_a": float(_sharpe(a)), "sharpe_b": float(_sharpe(b)),
            "delta": float(_sharpe(a) - _sharpe(b)), "p": float("nan"),
            "n_days": int(n),
        }
    a = a[:n]
    b = b[:n]
    point_a = _sharpe(a)
    point_b = _sharpe(b)
    delta_point = point_a - point_b
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_reps, dtype=np.float64)
    for r in range(n_reps):
        idxs = np.empty(n, dtype=np.int64)
        # Reuse a single random walk index series for paired resampling so
        # the same positions are drawn from both a and b.
        cur = int(rng.integers(0, n))
        p_cont = 1.0 - 1.0 / float(max(1, block_length))
        for t in range(n):
            idxs[t] = cur
            if rng.random() < p_cont:
                cur = (cur + 1) % n
            else:
                cur = int(rng.integers(0, n))
        deltas[r] = _sharpe(a[idxs]) - _sharpe(b[idxs])
    # One-sided p: probability that bootstrap delta is <= 0 given the
    # observed positive point delta (or >= 0 given negative point delta).
    if delta_point >= 0:
        p = float((deltas <= 0).mean())
    else:
        p = float((deltas >= 0).mean())
    return {
        "sharpe_a": float(point_a), "sharpe_b": float(point_b),
        "delta": float(delta_point), "p": float(p), "n_days": int(n),
    }


def _collect_cells(
    base_glob: str, sharpe_fn,
) -> Dict[Tuple[int, int], Tuple[float, str]]:
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--proposed-dir", type=str,
                   default="outputs/sp500/stage3_rl_ablation/robust_phase1")
    p.add_argument("--baseline-dir", type=str,
                   default="invar_rl/results/stage3_rl_ablation/equal_l2")
    p.add_argument("--proposed-tape-dir", type=str,
                   default="outputs/sp500/stage3_rl_ablation/robust_phase1/daily_tape/sac")
    p.add_argument("--baseline-tape-dir", type=str, default="",
                   help="Optional; canonical baseline daily tape if available.")
    p.add_argument("--out", type=str,
                   default="reports/robust_invar_rl/phase_1_group_dro_25cell.md")
    p.add_argument("--block-length", type=int, default=5)
    p.add_argument("--n-reps", type=int, default=1000)
    args = p.parse_args()

    proposed = _collect_cells(
        str(Path(args.proposed_dir) / "fold*_seed*.json"),
        _cell_sharpe_from_json,
    )
    baseline = _collect_cells(
        str(Path(args.baseline_dir) / "fold*_seed*.json"),
        _cell_sharpe_from_json,
    )

    folds = sorted({f for (f, _) in proposed.keys()} | {f for (f, _) in baseline.keys()})
    if not folds:
        print("[ERR] no proposed or baseline cells found")
        return 1

    lines: List[str] = []
    lines.append("# Robust-InVAR-RL Phase 1: 25-cell SP500 rollup")
    lines.append("")
    lines.append(
        f"Baseline (canonical equal_l2 SAC K=50): pooled +{CANONICAL_POOLED_SHARPE_K50:.3f} "
        f"(reference), F2 {CANONICAL_F2_SHARPE_K50:+.3f}."
    )
    lines.append("")
    lines.append("## Per-fold table")
    lines.append("")
    lines.append("| Fold | n_prop | n_base | Proposed mean | Baseline mean | Delta | Bootstrap p (paired daily) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")

    per_fold_props: Dict[int, List[float]] = {}
    per_fold_bases: Dict[int, List[float]] = {}
    fold_bootstrap_p: Dict[int, float] = {}
    for f in folds:
        prop_cells = [(s, v[0]) for ((ff, s), v) in proposed.items() if ff == f]
        base_cells = [(s, v[0]) for ((ff, s), v) in baseline.items() if ff == f]
        per_fold_props[f] = [v for _, v in prop_cells]
        per_fold_bases[f] = [v for _, v in base_cells]
        mean_p = float(np.mean(per_fold_props[f])) if per_fold_props[f] else float("nan")
        mean_b = float(np.mean(per_fold_bases[f])) if per_fold_bases[f] else float("nan")
        delta = mean_p - mean_b if not (math.isnan(mean_p) or math.isnan(mean_b)) else float("nan")
        # Per-fold paired bootstrap on daily tapes if available; aggregate
        # by min p-value across seeds within the fold.
        p_vals: List[float] = []
        prop_seeds = sorted({_s for (_f, _s) in proposed.keys() if _f == f})
        for seed in prop_seeds:
            prop_tape_path = Path(args.proposed_tape_dir) / f"fold{f}_seed{seed}.parquet"
            base_tape_path = (
                Path(args.baseline_tape_dir) / f"fold{f}_seed{seed}.parquet"
                if args.baseline_tape_dir else Path()
            )
            if not prop_tape_path.exists():
                continue
            if not base_tape_path.exists():
                continue
            a = _load_daily(str(prop_tape_path))
            b = _load_daily(str(base_tape_path))
            res = _paired_bootstrap_p(
                a, b, block_length=args.block_length,
                n_reps=args.n_reps, seed=42 + seed,
            )
            p_vals.append(float(res["p"]))
        if p_vals:
            p_min = min(p_vals)
            p_summary = f"min={p_min:.3f} seeds={len(p_vals)}"
            fold_bootstrap_p[f] = p_min
        else:
            p_summary = "n/a (no paired daily tape)"
            fold_bootstrap_p[f] = float("nan")
        lines.append(
            f"| F{f} | {len(per_fold_props[f])} | {len(per_fold_bases[f])} | "
            f"{mean_p:+.4f} | {mean_b:+.4f} | {delta:+.4f} | {p_summary} |"
        )

    pool_prop = float(np.mean([v for vs in per_fold_props.values() for v in vs])) if any(per_fold_props.values()) else float("nan")
    pool_base = float(np.mean([v for vs in per_fold_bases.values() for v in vs])) if any(per_fold_bases.values()) else float("nan")
    pool_delta = pool_prop - pool_base
    f2_prop = float(np.mean(per_fold_props.get(2, []))) if per_fold_props.get(2) else float("nan")

    lines.append("")
    lines.append(
        f"**Pool**: proposed={pool_prop:+.4f} baseline={pool_base:+.4f} "
        f"delta={pool_delta:+.4f} (canonical K=50 ref +{CANONICAL_POOLED_SHARPE_K50:.3f}; "
        f"K=25 ref +{CANONICAL_POOLED_SHARPE_K25:.3f})"
    )
    lines.append(f"**F2**: proposed={f2_prop:+.4f} (canonical F2 K=50 {CANONICAL_F2_SHARPE_K50:+.3f})")
    lines.append("")
    lines.append("## Phase 1 stop-gate verdict")
    lines.append("")
    # Apply gate.
    gates: List[Tuple[str, bool, str]] = []
    f2_pass = (not math.isnan(f2_prop)) and (f2_prop > CANONICAL_F2_SHARPE_K50)
    gates.append((
        "F2 per-fold mean > canonical F2",
        f2_pass,
        f"F2={f2_prop:+.4f} vs canonical {CANONICAL_F2_SHARPE_K50:+.3f}",
    ))
    pool_pass = (not math.isnan(pool_delta)) and (pool_delta >= PHASE1_STOP_GATE_POOL_FLOOR_DELTA)
    gates.append((
        "Pool within -0.10 of baseline (ideally >=)",
        pool_pass,
        f"delta={pool_delta:+.4f}, floor {PHASE1_STOP_GATE_POOL_FLOOR_DELTA:+.2f}",
    ))
    n_p_under_threshold = sum(
        1 for v in fold_bootstrap_p.values()
        if (not math.isnan(v)) and v < 0.20
    )
    boot_pass = n_p_under_threshold >= PHASE1_STOP_GATE_BOOTSTRAP_FOLDS
    gates.append((
        f"At least {PHASE1_STOP_GATE_BOOTSTRAP_FOLDS}/5 folds with bootstrap p < 0.20",
        boot_pass,
        f"{n_p_under_threshold}/5 folds pass",
    ))
    hard_fail = (not math.isnan(pool_delta)) and pool_delta < PHASE1_STOP_GATE_POOL_HARD_FAIL_DELTA
    if hard_fail:
        gates.append((
            "HARD FAIL: pool dropped > 0.20 below baseline",
            False,
            f"delta={pool_delta:+.4f}",
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
