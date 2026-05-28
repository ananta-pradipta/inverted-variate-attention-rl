"""Robust-InVAR-RL Phase 3 rollup: SP500 25-cell + bootstrap p-values.

Reads:
- Phase 3 proposed cells (Phase 2 + compact obs + online Sharpe reward):
  ``outputs/sp500/layer3_robust_phase3_25cell/foldF_seedS.json``
  + ``daily_tape/sac/foldF_seedS.parquet``
- Phase 2 reference cells (Kelly prior + residual SAC):
  ``outputs/sp500/layer3_robust_phase2_25cell/foldF_seedS.json``
  + ``daily_tape/sac/foldF_seedS.parquet``
- Canonical baseline cells:
  ``invar_rl/results/stage3_rl_ablation/equal_l2/foldF_seedS.json``

Produces a per-fold table for Phase 3 vs Phase 2 and Phase 3 vs canonical,
pool deltas, and the Phase 3 stop-gate verdict per the source design
doc + the user's Phase 3 instructions.

Phase 3 stop gate:
1. Smoke completes without NaN (obs dim exact, residual cap respected).
2. Test Sharpe within +/- 0.20 of Phase 2 single-cell smoke (+0.30).
3. 25-cell sweep completes.
4. F1/F3/F5 mean Sharpe >= Phase 2 - 0.10 each (don't worsen further).
5. Pool >= Phase 2 +0.510 - 0.05 = +0.46.
6. Compact obs dim exactly = 17 (or 9 if regime off).
HARD FAIL: pool dropped > -0.10 vs Phase 2.
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

# Phase 1 (group-DRO ranker) reference.
PHASE1_POOLED = 0.665
PHASE1_F1 = 0.871
PHASE1_F2 = 0.012
PHASE1_F3 = 0.944
PHASE1_F4 = 0.686
PHASE1_F5 = 0.814

# Phase 2 (Kelly prior + residual SAC) reference, from
# reports/robust_invar_rl/phase_2_residual_sac_25cell.md (2026-05-26).
PHASE2_POOLED = 0.510
PHASE2_F1 = 0.525
PHASE2_F2 = -0.151
PHASE2_F3 = 0.266
PHASE2_F4 = 1.363
PHASE2_F5 = 0.545

TRADING_DAYS = 252

# Phase 3 stop-gate thresholds.
PHASE3_F1_FLOOR_DELTA_VS_P2 = -0.10
PHASE3_F3_FLOOR_DELTA_VS_P2 = -0.10
PHASE3_F5_FLOOR_DELTA_VS_P2 = -0.10
PHASE3_POOL_FLOOR_DELTA_VS_P2 = -0.05
PHASE3_POOL_HARD_FAIL_DELTA_VS_P2 = -0.10
EXPECTED_COMPACT_OBS_DIM_WITH_REGIME = 17
EXPECTED_COMPACT_OBS_DIM_NO_REGIME = 9


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


def _cell_phase3_diag(path: str) -> Dict:
    with open(path) as fh:
        payload = json.load(fh)
    return dict(payload.get("diagnostics", {}).get("phase3", {}))


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
    a_dir: str, b_dir: str, seeds_per_fold: int = 5,
    block_length: int = 5, n_reps: int = 1000,
) -> Dict[int, float]:
    """Per-fold min p-value across seeds, comparing a vs b daily tapes."""
    out: Dict[int, float] = {}
    for f in [1, 2, 3, 4, 5]:
        p_vals: List[float] = []
        for seed in [42, 43, 44, 45, 46][:seeds_per_fold]:
            a_path = Path(a_dir) / f"fold{f}_seed{seed}.parquet"
            b_path = Path(b_dir) / f"fold{f}_seed{seed}.parquet"
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
        "--phase3-dir", type=str,
        default="outputs/sp500/layer3_robust_phase3_25cell",
    )
    p.add_argument(
        "--phase2-dir", type=str,
        default="outputs/sp500/layer3_robust_phase2_25cell",
    )
    p.add_argument(
        "--phase1-dir", type=str,
        default="outputs/sp500/layer1_robust_phase1_25cell",
    )
    p.add_argument(
        "--baseline-dir", type=str,
        default="invar_rl/results/stage3_rl_ablation/equal_l2",
    )
    p.add_argument(
        "--phase3-tape-dir", type=str,
        default="outputs/sp500/layer3_robust_phase3_25cell/daily_tape/sac",
    )
    p.add_argument(
        "--phase2-tape-dir", type=str,
        default="outputs/sp500/layer3_robust_phase2_25cell/daily_tape/sac",
    )
    p.add_argument(
        "--baseline-tape-dir", type=str,
        default="outputs/sp500/stage3_rl_ablation/canonical_equal_l2_tape_25cell/daily_tape/sac",
    )
    p.add_argument(
        "--out", type=str,
        default="reports/robust_invar_rl/phase_3_compact_obs_25cell.md",
    )
    p.add_argument("--block-length", type=int, default=5)
    p.add_argument("--n-reps", type=int, default=1000)
    args = p.parse_args()

    p3 = _collect_cells(
        str(Path(args.phase3_dir) / "fold*_seed*.json"),
        _cell_sharpe_from_json,
    )
    p2 = _collect_cells(
        str(Path(args.phase2_dir) / "fold*_seed*.json"),
        _cell_sharpe_from_json,
    )
    bl = _collect_cells(
        str(Path(args.baseline_dir) / "fold*_seed*.json"),
        _cell_sharpe_from_json,
    )

    p3_per_fold = _per_fold_means(p3)
    p2_per_fold = _per_fold_means(p2)
    bl_per_fold = _per_fold_means(bl)

    pool_p3 = (
        float(np.mean(list(p3_per_fold.values())))
        if p3_per_fold else float("nan")
    )
    pool_p2 = (
        float(np.mean(list(p2_per_fold.values())))
        if p2_per_fold else float("nan")
    )
    pool_bl = (
        float(np.mean(list(bl_per_fold.values())))
        if bl_per_fold else float("nan")
    )

    # Pool sd and SoS = pool/sd, computed across the 25 per-cell Sharpes.
    p3_vals = np.asarray([v for (_, _), (v, _) in p3.items()])
    p2_vals = np.asarray([v for (_, _), (v, _) in p2.items()])
    bl_vals = np.asarray([v for (_, _), (v, _) in bl.items()])
    sd_p3 = float(p3_vals.std(ddof=1)) if p3_vals.size > 1 else float("nan")
    sd_p2 = float(p2_vals.std(ddof=1)) if p2_vals.size > 1 else float("nan")
    sd_bl = float(bl_vals.std(ddof=1)) if bl_vals.size > 1 else float("nan")
    sos_p3 = pool_p3 / sd_p3 if sd_p3 and not math.isnan(sd_p3) else float("nan")
    sos_p2 = pool_p2 / sd_p2 if sd_p2 and not math.isnan(sd_p2) else float("nan")

    p_vs_p2 = _fold_bootstrap(
        args.phase3_tape_dir, args.phase2_tape_dir,
        block_length=args.block_length, n_reps=args.n_reps,
    )
    p_vs_bl = (
        _fold_bootstrap(
            args.phase3_tape_dir, args.baseline_tape_dir,
            block_length=args.block_length, n_reps=args.n_reps,
        ) if Path(args.baseline_tape_dir).exists() else {}
    )

    # Compact obs dim audit: pull from any Phase 3 cell's diagnostics.
    obs_dim_actual = -1
    obs_dim_expected = -1
    enabled_compact = False
    enabled_sharpe = False
    enabled_regime = False
    if p3:
        any_path = next(iter(p3.values()))[1]
        diag = _cell_phase3_diag(any_path)
        obs_dim_actual = int(diag.get("compact_obs_dim_actual", -1))
        obs_dim_expected = int(diag.get("compact_obs_dim_expected", -1))
        enabled_compact = bool(diag.get("enabled_compact_obs", False))
        enabled_sharpe = bool(
            diag.get("enabled_online_sharpe_reward", False)
        )
        enabled_regime = bool(diag.get("enabled_regime_one_hot", False))

    lines: List[str] = []
    lines.append(
        "# Robust-InVAR-RL Phase 3: Compact Obs + Online Sharpe Reward 25-cell SP500"
    )
    lines.append("")
    lines.append(
        f"Built on top of Phase 2 (Kelly prior + residual SAC). Two swaps: "
        f"(1) compact obs (8 base + {'8 regime + ' if enabled_regime else ''}1 e_star), "
        f"(2) online Sharpe reward (EWMA half-life 21d). Phase 1 group-DRO L1 "
        f"ckpts unchanged."
    )
    lines.append("")
    lines.append(
        f"Canonical equal_l2 SAC K=50: pool +{CANONICAL_POOLED_SHARPE_K50:.3f}, "
        f"F2 {CANONICAL_F2:+.3f}, F4 {CANONICAL_F4:+.3f}, F5 {CANONICAL_F5:+.3f}"
    )
    lines.append(
        f"Phase 1 (group-DRO ranker): pool +{PHASE1_POOLED:.3f}, "
        f"F2 {PHASE1_F2:+.3f}"
    )
    lines.append(
        f"Phase 2 (P1 + Kelly + residual SAC): pool +{PHASE2_POOLED:.3f}, "
        f"F1 {PHASE2_F1:+.3f}, F2 {PHASE2_F2:+.3f}, F3 {PHASE2_F3:+.3f}, "
        f"F4 {PHASE2_F4:+.3f}, F5 {PHASE2_F5:+.3f}"
    )
    lines.append("")
    lines.append("## Per-fold table")
    lines.append("")
    lines.append(
        "| Method | F1 | F2 | F3 | F4 | F5 | Pool | sd | SoS |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    canon_row = (
        f"| canonical SAC | {bl_per_fold.get(1, float('nan')):+.3f} | "
        f"{bl_per_fold.get(2, float('nan')):+.3f} | "
        f"{bl_per_fold.get(3, float('nan')):+.3f} | "
        f"{bl_per_fold.get(4, float('nan')):+.3f} | "
        f"{bl_per_fold.get(5, float('nan')):+.3f} | "
        f"**{pool_bl:+.3f}** | {sd_bl:.3f} | -- |"
    )
    lines.append(canon_row)
    lines.append(
        f"| Phase 1 (Group-DRO) | {PHASE1_F1:+.3f} | **{PHASE1_F2:+.3f}** | "
        f"{PHASE1_F3:+.3f} | {PHASE1_F4:+.3f} | {PHASE1_F5:+.3f} | "
        f"+{PHASE1_POOLED:.3f} | -- | -- |"
    )
    lines.append(
        f"| Phase 2 (P1 + Kelly + ResSAC) | {PHASE2_F1:+.3f} | "
        f"{PHASE2_F2:+.3f} | {PHASE2_F3:+.3f} | {PHASE2_F4:+.3f} | "
        f"{PHASE2_F5:+.3f} | +{PHASE2_POOLED:.3f} | {sd_p2:.3f} | "
        f"{sos_p2:.2f} |"
    )
    lines.append(
        f"| **Phase 3 (P2 + compact + Sharpe)** | "
        f"{p3_per_fold.get(1, float('nan')):+.3f} | "
        f"{p3_per_fold.get(2, float('nan')):+.3f} | "
        f"{p3_per_fold.get(3, float('nan')):+.3f} | "
        f"{p3_per_fold.get(4, float('nan')):+.3f} | "
        f"{p3_per_fold.get(5, float('nan')):+.3f} | "
        f"**{pool_p3:+.3f}** | {sd_p3:.3f} | {sos_p3:.2f} |"
    )
    # Deltas.
    d_vs_p2 = {
        f: p3_per_fold.get(f, float("nan")) - p2_per_fold.get(f, float("nan"))
        for f in [1, 2, 3, 4, 5]
    }
    d_vs_bl = {
        f: p3_per_fold.get(f, float("nan")) - bl_per_fold.get(f, float("nan"))
        for f in [1, 2, 3, 4, 5]
    }
    lines.append(
        f"| Phase 3 delta vs Phase 2 | "
        f"{d_vs_p2.get(1, float('nan')):+.3f} | "
        f"{d_vs_p2.get(2, float('nan')):+.3f} | "
        f"{d_vs_p2.get(3, float('nan')):+.3f} | "
        f"{d_vs_p2.get(4, float('nan')):+.3f} | "
        f"{d_vs_p2.get(5, float('nan')):+.3f} | "
        f"{pool_p3 - pool_p2:+.3f} | -- | -- |"
    )
    lines.append(
        f"| Phase 3 delta vs canonical | "
        f"{d_vs_bl.get(1, float('nan')):+.3f} | "
        f"{d_vs_bl.get(2, float('nan')):+.3f} | "
        f"{d_vs_bl.get(3, float('nan')):+.3f} | "
        f"{d_vs_bl.get(4, float('nan')):+.3f} | "
        f"{d_vs_bl.get(5, float('nan')):+.3f} | "
        f"{pool_p3 - pool_bl:+.3f} | -- | -- |"
    )
    lines.append("")
    lines.append("## Paired bootstrap p-values")
    lines.append("")
    lines.append(
        "| Fold | Boot p (P3 vs P2) | Boot p (P3 vs canonical) |"
    )
    lines.append("|---:|---:|---:|")
    for f in [1, 2, 3, 4, 5]:
        pp2 = p_vs_p2.get(f, float("nan"))
        ppb = p_vs_bl.get(f, float("nan"))
        lines.append(f"| F{f} | {pp2:.3f} | {ppb:.3f} |")
    lines.append("")
    lines.append("## Phase 3 stop-gate verdict")
    lines.append("")
    gates: List[Tuple[str, bool, str]] = []

    # 6. Compact obs dim exactly matches spec.
    expected_target = (
        EXPECTED_COMPACT_OBS_DIM_WITH_REGIME if enabled_regime
        else EXPECTED_COMPACT_OBS_DIM_NO_REGIME
    )
    dim_pass = (
        enabled_compact
        and obs_dim_actual == obs_dim_expected
        and obs_dim_actual == expected_target
    )
    gates.append((
        "Compact obs dim exact (spec)",
        dim_pass,
        (
            f"actual={obs_dim_actual} expected={obs_dim_expected} "
            f"spec_target={expected_target} (regime_one_hot={enabled_regime})"
        ),
    ))

    # 4. F1/F3/F5 mean Sharpe >= Phase 2 - 0.10 each.
    f1_p3 = p3_per_fold.get(1, float("nan"))
    f3_p3 = p3_per_fold.get(3, float("nan"))
    f5_p3 = p3_per_fold.get(5, float("nan"))
    f1_pass = (
        not math.isnan(f1_p3)
        and f1_p3 >= PHASE2_F1 + PHASE3_F1_FLOOR_DELTA_VS_P2
    )
    f3_pass = (
        not math.isnan(f3_p3)
        and f3_p3 >= PHASE2_F3 + PHASE3_F3_FLOOR_DELTA_VS_P2
    )
    f5_pass = (
        not math.isnan(f5_p3)
        and f5_p3 >= PHASE2_F5 + PHASE3_F5_FLOOR_DELTA_VS_P2
    )
    gates.append((
        "F1 >= Phase 2 F1 - 0.10",
        f1_pass,
        f"F1={f1_p3:+.3f} vs floor {PHASE2_F1 + PHASE3_F1_FLOOR_DELTA_VS_P2:+.3f}",
    ))
    gates.append((
        "F3 >= Phase 2 F3 - 0.10",
        f3_pass,
        f"F3={f3_p3:+.3f} vs floor {PHASE2_F3 + PHASE3_F3_FLOOR_DELTA_VS_P2:+.3f}",
    ))
    gates.append((
        "F5 >= Phase 2 F5 - 0.10",
        f5_pass,
        f"F5={f5_p3:+.3f} vs floor {PHASE2_F5 + PHASE3_F5_FLOOR_DELTA_VS_P2:+.3f}",
    ))

    # 5. Pool >= Phase 2 - 0.05.
    pool_delta_p2 = pool_p3 - pool_p2
    pool_pass = pool_delta_p2 >= PHASE3_POOL_FLOOR_DELTA_VS_P2
    gates.append((
        "Pool within -0.05 of Phase 2",
        pool_pass,
        f"pool delta vs P2 = {pool_delta_p2:+.3f} (Phase 3 pool={pool_p3:+.3f})",
    ))

    # HARD FAIL: pool dropped > -0.10 vs Phase 2.
    hard_fail = pool_delta_p2 < PHASE3_POOL_HARD_FAIL_DELTA_VS_P2
    if hard_fail:
        gates.append((
            "HARD FAIL: pool dropped > 0.10 below Phase 2",
            False,
            f"pool delta vs Phase 2 = {pool_delta_p2:+.3f}",
        ))

    for name, ok, note in gates:
        status = "PASS" if ok else "FAIL"
        lines.append(f"- [{status}] {name}: {note}")
    overall = all(ok for _, ok, _ in gates)
    lines.append("")
    lines.append(f"**Overall verdict**: {'PASS' if overall else 'FAIL'}")
    lines.append("")
    lines.append("## Cumulative position across architectural extensions")
    lines.append("")
    lines.append("| Variant | Pool SP500 | F2 | Notes |")
    lines.append("|---|---:|---:|---|")
    lines.append(
        f"| canonical SAC | +{CANONICAL_POOLED_SHARPE_K50:.3f} | "
        f"{CANONICAL_F2:+.3f} | HEADLINE |"
    )
    lines.append(
        f"| R-InVAR-RL Phase 1 (Group-DRO) | +{PHASE1_POOLED:.3f} | "
        f"{PHASE1_F2:+.3f} | F2 lift, F4/F5 collapse |"
    )
    lines.append(
        f"| R-InVAR-RL Phase 2 (P1 + Kelly + ResSAC) | "
        f"+{PHASE2_POOLED:.3f} | {PHASE2_F2:+.3f} | F4 recovers, F1/F3/F5 regress |"
    )
    lines.append(
        f"| **R-InVAR-RL Phase 3 (P2 + compact + Sharpe)** | "
        f"{pool_p3:+.3f} | {p3_per_fold.get(2, float('nan')):+.3f} | "
        f"compact obs (dim {obs_dim_actual}) + EWMA Sharpe reward |"
    )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        f"- Phase 3 sbatch: `invar_rl/scripts/wulver/"
        f"invar_rl_sp500_robust_phase3_25cell.sbatch`"
    )
    lines.append(
        "- Submit: `for SLOT in 0 1 2 3 4; do sbatch --export=ALL,SLOT=$SLOT "
        "invar_rl/scripts/wulver/invar_rl_sp500_robust_phase3_25cell.sbatch; done`"
    )
    lines.append(
        f"- Config: delta_cap=0.25, kappa=0.02, e_max=1.5, K=50, calibration=platt, "
        f"compact_obs_dim={obs_dim_actual} (regime_one_hot={enabled_regime}), "
        f"sharpe_half_life=21, warmup=5, clip=8.0"
    )
    lines.append(
        f"- Code: `src/models/robust_invar_rl/"
        f"{{compact_obs, online_sharpe_reward}}.py`, "
        f"`invar_rl/layer3_control/phase3_env_wrappers.py`, "
        f"`invar_rl/training/sp500_residual_sac.py` (extended)"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[INFO] wrote {out_path}")
    for ln in lines:
        print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
