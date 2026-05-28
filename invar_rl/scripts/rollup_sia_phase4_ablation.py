"""Roll up the InVAR-RL-SIA Phase 4 SP500 ablation matrix.

Reads per-cell summary jsons for two Phase 4 ablations:

  - no_a: outputs/sp500/layer2_sia/phase4_no_a/summary/foldF_seedS.json
  - no_s: outputs/sp500/layer2_sia/phase4_no_s/summary/foldF_seedS.json

Compares against:

  - full_sia: outputs/sp500/layer2_sia/phase2_regime_beta_1e-4/summary/...
    (S+I+A, lambda_inv=0.1, regime_label=ON, beta_kl=1e-4, 25 cells)
  - no_i: F1-only mini-sweep mean +0.329 (lambda_inv=0; PARTIAL reference;
    user-provided from the lambda mini-sweep)
  - canonical SAC: +0.945 pooled (from reports/_rollup_fair_k25.json)

Outputs:

  - Per-variant pooled mean, sd, sem across the 25 (or 5) cells.
  - Per-fold mean +/- sd table for every variant.
  - Per-component lift attribution:
      S contribution = full_sia - no_s
      A contribution = full_sia - no_a
      I contribution = (full_sia F1 mean) - (no_i F1 mean)  [partial, F1 only]
  - Variance comparison across variants.
  - JSON dump for downstream consumption.

Usage::

    python -m invar_rl.scripts.rollup_sia_phase4_ablation \
        --no-a-root outputs/sp500/layer2_sia/phase4_no_a \
        --no-s-root outputs/sp500/layer2_sia/phase4_no_s \
        --full-sia-root outputs/sp500/layer2_sia/phase2_regime_beta_1e-4 \
        --no-i-f1-mean 0.329 \
        --canonical-pooled 0.945 \
        --json-out reports/sia/_phase4_ablation_rollup.json
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


def _load_cell(summary_dir: Path, fold: int, seed: int) -> Optional[dict]:
    p = summary_dir / f"fold{fold}_seed{seed}.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def _safe_sd(xs: List[float]) -> float:
    return float(pstdev(xs)) if len(xs) > 1 else 0.0


def _load_variant_matrix(
    summary_dir: Path,
) -> Tuple[
    Dict[Tuple[int, int], Optional[float]], List[Tuple[int, int]]
]:
    matrix: Dict[Tuple[int, int], Optional[float]] = {}
    missing: List[Tuple[int, int]] = []
    for f in _FOLDS:
        for s in _SEEDS:
            payload = _load_cell(summary_dir, f, s)
            if payload is None:
                matrix[(f, s)] = None
                missing.append((f, s))
            else:
                matrix[(f, s)] = float(payload["test_pooled_sharpe"])
    return matrix, missing


def _summarise_variant(
    matrix: Dict[Tuple[int, int], Optional[float]],
) -> Dict[str, object]:
    pooled_cells: List[float] = []
    fold_means: Dict[int, float] = {}
    fold_sds: Dict[int, float] = {}
    for f in _FOLDS:
        cells = [matrix[(f, s)] for s in _SEEDS]
        present = [c for c in cells if c is not None]
        pooled_cells.extend(present)
        fold_means[f] = mean(present) if present else float("nan")
        fold_sds[f] = _safe_sd(present)
    pooled_mean = mean(pooled_cells) if pooled_cells else float("nan")
    pooled_sd = _safe_sd(pooled_cells)
    pooled_sem = (
        pooled_sd / math.sqrt(len(pooled_cells)) if pooled_cells else 0.0
    )
    return {
        "n_cells": len(pooled_cells),
        "pooled_mean": pooled_mean,
        "pooled_sd": pooled_sd,
        "pooled_sem": pooled_sem,
        "per_fold_mean": {str(f): fold_means[f] for f in _FOLDS},
        "per_fold_sd": {str(f): fold_sds[f] for f in _FOLDS},
        "per_cell": {
            f"fold{f}_seed{s}": matrix[(f, s)]
            for f in _FOLDS for s in _SEEDS
        },
    }


def _print_per_cell_table(
    label: str, matrix: Dict[Tuple[int, int], Optional[float]]
) -> None:
    print(f"## {label}")
    print()
    header = "fold | " + " | ".join(f"seed {s}" for s in _SEEDS) + " | mean +/- sd"
    print(header)
    print("-" * len(header))
    for f in _FOLDS:
        cells = [matrix[(f, s)] for s in _SEEDS]
        present = [c for c in cells if c is not None]
        m = mean(present) if present else float("nan")
        sd = _safe_sd(present)
        row_cells = [
            f"{c:+.3f}" if c is not None else "MISS" for c in cells
        ]
        print(f"F{f}   | " + " | ".join(row_cells) + f" | {m:+.3f} +/- {sd:.3f}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--no-a-root", type=Path,
        default=Path("outputs/sp500/layer2_sia/phase4_no_a"),
    )
    ap.add_argument(
        "--no-s-root", type=Path,
        default=Path("outputs/sp500/layer2_sia/phase4_no_s"),
    )
    ap.add_argument(
        "--no-i-root", type=Path,
        default=Path("outputs/sp500/layer2_sia/phase4_no_i"),
        help=(
            "Full 25-cell no_i (lambda_inv=0.0) summary root. If missing, "
            "fall back to the F1 mini-sweep mean reference."
        ),
    )
    ap.add_argument(
        "--full-sia-root", type=Path,
        default=Path("outputs/sp500/layer2_sia/phase2_regime_beta_1e-4"),
    )
    ap.add_argument(
        "--no-i-f1-mean", type=float, default=0.329,
        help=(
            "Fallback F1-only mean of the lambda_inv=0 mini-sweep (no_i). "
            "Used only when --no-i-root is missing or empty."
        ),
    )
    ap.add_argument(
        "--canonical-pooled", type=float, default=0.945,
        help="Canonical SAC SP500 25-cell pooled Sharpe reference.",
    )
    ap.add_argument(
        "--json-out", type=Path,
        default=Path("reports/sia/_phase4_ablation_rollup.json"),
    )
    args = ap.parse_args()

    full_dir = args.full_sia_root / "summary"
    no_a_dir = args.no_a_root / "summary"
    no_s_dir = args.no_s_root / "summary"
    no_i_dir = args.no_i_root / "summary"
    for d in (full_dir, no_a_dir, no_s_dir):
        if not d.exists():
            raise SystemExit(f"summary dir missing: {d}")

    full_matrix, full_missing = _load_variant_matrix(full_dir)
    no_a_matrix, no_a_missing = _load_variant_matrix(no_a_dir)
    no_s_matrix, no_s_missing = _load_variant_matrix(no_s_dir)

    full_summary = _summarise_variant(full_matrix)
    no_a_summary = _summarise_variant(no_a_matrix)
    no_s_summary = _summarise_variant(no_s_matrix)

    no_i_available = no_i_dir.exists() and any(no_i_dir.glob("fold*_seed*.json"))
    if no_i_available:
        no_i_matrix, no_i_missing = _load_variant_matrix(no_i_dir)
        no_i_summary = _summarise_variant(no_i_matrix)
    else:
        no_i_matrix = {}
        no_i_missing = []
        no_i_summary = None

    print("# InVAR-RL-SIA Phase 4 SP500 ablation matrix rollup")
    print()
    print("Sources:")
    print(f"  full_sia: {full_dir}")
    print(f"  no_a:     {no_a_dir}")
    print(f"  no_s:     {no_s_dir}")
    if no_i_available:
        print(f"  no_i:     {no_i_dir}  (full 25-cell, lambda_inv=0.0)")
    else:
        print(f"  no_i:     F1 mini-sweep mean = {args.no_i_f1_mean:+.4f} (PARTIAL)")
    print(f"  canonical SAC pooled: {args.canonical_pooled:+.4f}")
    print()

    if full_missing or no_a_missing or no_s_missing or no_i_missing:
        print(
            f"MISSING cells -- full_sia: {len(full_missing)}, "
            f"no_a: {len(no_a_missing)}, no_s: {len(no_s_missing)}, "
            f"no_i: {len(no_i_missing) if no_i_available else 'NA'}"
        )
        print()

    # 4-variant pooled mean table.
    print("## 4-variant pooled mean comparison")
    print()
    hdr = (
        "variant      | n  | pooled mean | pooled sd | pooled sem | "
        "delta_vs_full | delta_vs_canon"
    )
    print(hdr)
    print("-" * len(hdr))
    full_pool = float(full_summary["pooled_mean"])
    rows = [
        ("full_sia", full_summary, 0.0),
        ("no_a    ", no_a_summary, float(no_a_summary["pooled_mean"]) - full_pool),
        ("no_s    ", no_s_summary, float(no_s_summary["pooled_mean"]) - full_pool),
    ]
    if no_i_available:
        rows.append(
            ("no_i    ", no_i_summary, float(no_i_summary["pooled_mean"]) - full_pool)
        )
    for label, s, d_full in rows:
        delta_canon = float(s["pooled_mean"]) - float(args.canonical_pooled)
        print(
            f"{label} | {int(s['n_cells']):2d} | {float(s['pooled_mean']):+11.4f} | "
            f"{float(s['pooled_sd']):9.4f} | {float(s['pooled_sem']):10.4f} | "
            f"{d_full:+13.4f} | {delta_canon:+14.4f}"
        )
    if not no_i_available:
        print(
            f"no_i (F1 only) | 5 | {args.no_i_f1_mean:+11.4f} | "
            "       NA |         NA |             NA |             NA  "
            "(PARTIAL: lambda_inv=0 reference, F1 cells only)"
        )
    print()

    # Per-fold breakdown per variant.
    print("## Per-fold mean Sharpe table")
    print()
    if no_i_available:
        hdr2 = (
            "fold | full_sia          | no_a              | no_s              "
            "| no_i"
        )
    else:
        hdr2 = "fold | full_sia          | no_a              | no_s"
    print(hdr2)
    print("-" * len(hdr2))
    for f in _FOLDS:
        sf = float(full_summary["per_fold_mean"][str(f)])
        sf_sd = float(full_summary["per_fold_sd"][str(f)])
        na = float(no_a_summary["per_fold_mean"][str(f)])
        na_sd = float(no_a_summary["per_fold_sd"][str(f)])
        ns = float(no_s_summary["per_fold_mean"][str(f)])
        ns_sd = float(no_s_summary["per_fold_sd"][str(f)])
        row = (
            f"F{f}   | {sf:+.3f} +/- {sf_sd:.3f} | "
            f"{na:+.3f} +/- {na_sd:.3f} | {ns:+.3f} +/- {ns_sd:.3f}"
        )
        if no_i_available:
            ni = float(no_i_summary["per_fold_mean"][str(f)])
            ni_sd = float(no_i_summary["per_fold_sd"][str(f)])
            row += f" | {ni:+.3f} +/- {ni_sd:.3f}"
        print(row)
    print()

    _print_per_cell_table("Per-cell table: full_sia", full_matrix)
    _print_per_cell_table("Per-cell table: no_a", no_a_matrix)
    _print_per_cell_table("Per-cell table: no_s", no_s_matrix)
    if no_i_available:
        _print_per_cell_table("Per-cell table: no_i", no_i_matrix)

    # Per-component lift attribution.
    print("## Per-component lift attribution (vs full SIA)")
    print()
    s_contrib = full_pool - float(no_s_summary["pooled_mean"])
    a_contrib = full_pool - float(no_a_summary["pooled_mean"])
    full_f1 = float(full_summary["per_fold_mean"]["1"])
    full_f2 = float(full_summary["per_fold_mean"]["2"])
    print(
        f"  S (sparse gates):       lift = full - no_s = "
        f"{full_pool:+.4f} - {float(no_s_summary['pooled_mean']):+.4f} = "
        f"{s_contrib:+.4f}"
    )
    print(
        f"  A (asymmetric critic):  lift = full - no_a = "
        f"{full_pool:+.4f} - {float(no_a_summary['pooled_mean']):+.4f} = "
        f"{a_contrib:+.4f}"
    )
    if no_i_available:
        i_contrib = full_pool - float(no_i_summary["pooled_mean"])
        i_f2 = full_f2 - float(no_i_summary["per_fold_mean"]["2"])
        print(
            f"  I (regime invariance):  lift = full - no_i = "
            f"{full_pool:+.4f} - {float(no_i_summary['pooled_mean']):+.4f} = "
            f"{i_contrib:+.4f}   [FULL 25-cell]"
        )
        print(
            f"    F2 lift (full_F2 - no_i_F2) = {full_f2:+.4f} - "
            f"{float(no_i_summary['per_fold_mean']['2']):+.4f} = {i_f2:+.4f}"
        )
        i_contrib_f1 = full_f1 - float(no_i_summary["per_fold_mean"]["1"])
    else:
        i_contrib = float("nan")
        i_contrib_f1 = full_f1 - float(args.no_i_f1_mean)
        i_f2 = float("nan")
        print(
            f"  I (regime invariance):  lift = full_F1 - no_i_F1 = "
            f"{full_f1:+.4f} - {float(args.no_i_f1_mean):+.4f} = "
            f"{i_contrib_f1:+.4f}   [PARTIAL, F1 only]"
        )
    print()

    # Variance comparison.
    print("## Variance comparison")
    print()
    print(
        f"  full_sia pooled sd: {float(full_summary['pooled_sd']):.4f}"
    )
    print(
        f"  no_a    pooled sd: {float(no_a_summary['pooled_sd']):.4f}  "
        f"(delta vs full: {float(no_a_summary['pooled_sd']) - float(full_summary['pooled_sd']):+.4f})"
    )
    print(
        f"  no_s    pooled sd: {float(no_s_summary['pooled_sd']):.4f}  "
        f"(delta vs full: {float(no_s_summary['pooled_sd']) - float(full_summary['pooled_sd']):+.4f})"
    )
    if no_i_available:
        print(
            f"  no_i    pooled sd: {float(no_i_summary['pooled_sd']):.4f}  "
            f"(delta vs full: {float(no_i_summary['pooled_sd']) - float(full_summary['pooled_sd']):+.4f})"
        )
    print()

    payload = {
        "sources": {
            "full_sia": str(full_dir),
            "no_a": str(no_a_dir),
            "no_s": str(no_s_dir),
        },
        "no_i_f1_mean_reference": float(args.no_i_f1_mean),
        "canonical_pooled_reference": float(args.canonical_pooled),
        "variants": {
            "full_sia": full_summary,
            "no_a": no_a_summary,
            "no_s": no_s_summary,
            "no_i": no_i_summary,
        },
        "missing": {
            "full_sia": [list(t) for t in full_missing],
            "no_a": [list(t) for t in no_a_missing],
            "no_s": [list(t) for t in no_s_missing],
            "no_i": [list(t) for t in no_i_missing] if no_i_available else [],
        },
        "component_lifts": {
            "S_sparse_gates": s_contrib,
            "A_asymmetric_critic": a_contrib,
            "I_regime_invariance_pooled": i_contrib,
            "I_regime_invariance_F2": i_f2,
            "I_regime_invariance_F1": i_contrib_f1,
        },
    }
    if no_i_available:
        payload["sources"]["no_i"] = str(no_i_dir)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
