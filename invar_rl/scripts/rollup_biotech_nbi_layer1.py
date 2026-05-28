"""Rollup the biotech NBI Phase 3 Layer 1 sweep results.

Reads every ``outputs/biotech_nbi/layer1/metrics/fold{F}_seed{S}.json``
written by ``invar_rl/training/biotech_nbi_layer1_eval.py``, computes
the per-fold pooled rank IC (mean and std across seeds), the all-25-cell
pooled rank IC (mean and std), and the canonical-vs-S&P-500 (and
parked NASDAQ-100 / DJIA-30) comparison. Patches the numbers into
``reports/biotech_nbi/phase_3_layer1_ic.md`` (replacing the TBD rows)
and prints a CLI summary.

Usage::

    PYTHONPATH=. python3 -m invar_rl.scripts.rollup_biotech_nbi_layer1
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np


METRICS_DIR_DEFAULT = Path("outputs/biotech_nbi/layer1/metrics")
REPORT_PATH_DEFAULT = Path("reports/biotech_nbi/phase_3_layer1_ic.md")

# Canonical S&P 500 InVAR pooled rank IC (locked 2026-05-19, Phase 3
# spec reference), parked NASDAQ-100 (Phase 3 reference), and parked
# DJIA-30 (Phase 3 reference, see reports/djia30/phase_3_layer1_ic.md).
SP500_POOLED_RANK_IC = 0.0278
SP500_POOLED_STD = 0.019
NASDAQ100_POOLED_RANK_IC = 0.0270
DJIA30_POOLED_RANK_IC = 0.0177

FOLD_LABELS = {
    1: "F1 covid 2020         ",
    2: "F2 rate-stress 2021-22",
    3: "F3 post-stress 2022-23",
    4: "F4 ai-rally 2024      ",
    5: "F5 fed-cut 2025-H2    ",
}


def _read_cells(metrics_dir: Path) -> List[dict]:
    cells = []
    for fp in sorted(metrics_dir.glob("fold*_seed*.json")):
        try:
            with open(fp) as fh:
                cells.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue
    return cells


def _per_fold_rollup(cells: List[dict]) -> Dict[int, Dict[str, float]]:
    """Return {fold: {'mean': ..., 'std': ..., 'n': ...}} for rank IC."""
    out: Dict[int, Dict[str, float]] = {}
    for f in [1, 2, 3, 4, 5]:
        vals = [
            c["test_rank_ic"] for c in cells
            if c["fold"] == f and "test_rank_ic" in c
        ]
        if not vals:
            out[f] = {"mean": float("nan"), "std": float("nan"), "n": 0}
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[f] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "n": int(arr.size),
        }
    return out


def _pooled(cells: List[dict]) -> Dict[str, float]:
    vals = [c["test_rank_ic"] for c in cells if "test_rank_ic" in c]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def _check_acceptance(per_fold: Dict[int, Dict[str, float]],
                      pooled: Dict[str, float],
                      cells: List[dict]) -> Dict[str, str]:
    f2 = per_fold.get(2, {"mean": float("nan")})
    f2_ok = (f2["mean"] >= 0.0) if not np.isnan(f2["mean"]) else False
    pool_ok = abs(pooled["mean"] - SP500_POOLED_RANK_IC) <= 0.3 * SP500_POOLED_RANK_IC \
        if not np.isnan(pooled["mean"]) else False
    nan_free = all(
        not (np.isnan(c.get("test_rank_ic", float("nan")))
             or np.isnan(c.get("val_rank_ic", float("nan"))))
        for c in cells
    )
    return {
        "f2_non_negative": "yes" if f2_ok else "no",
        "pooled_within_30pct_sp500": "yes" if pool_ok else "no",
        "nan_free": "yes" if nan_free else "no",
    }


def _patch_report(
    report_path: Path,
    per_fold: Dict[int, Dict[str, float]],
    pooled: Dict[str, float],
    acceptance: Dict[str, str],
) -> None:
    if not report_path.exists():
        print(f"[rollup] WARN report {report_path} missing; skipping patch")
        return
    text = report_path.read_text()

    # Per-fold table.
    pf_lines = []
    for f in [1, 2, 3, 4, 5]:
        d = per_fold[f]
        if d["n"]:
            pf_lines.append(
                f"| {FOLD_LABELS[f]} | {d['mean']:+.4f} | "
                f"{d['std']:.4f} | {d['n']} |"
            )
        else:
            pf_lines.append(
                f"| {FOLD_LABELS[f]} | TBD | TBD | TBD |"
            )
    pf_block = (
        "## Per-fold pooled rank IC (5 seeds, 25-run sweep)\n\n"
        "| Fold | Mean | Std | n |\n"
        "|------|------|-----|---|\n"
        + "\n".join(pf_lines)
        + "\n"
    )
    text = re.sub(
        r"## Per-fold pooled rank IC.*?(?=\n## )",
        pf_block + "\n",
        text,
        flags=re.DOTALL,
    )

    # Pooled table.
    pooled_block = (
        "## Pooled across all 25 cells\n\n"
        "| Mean | Std |\n"
        "|------|-----|\n"
        f"| {pooled['mean']:+.4f}  | {pooled['std']:.4f} |\n"
    )
    text = re.sub(
        r"## Pooled across all 25 cells.*?(?=\n## )",
        pooled_block + "\n",
        text,
        flags=re.DOTALL,
    )

    # Comparison table.
    ratio_sp500 = (
        f"{(pooled['mean'] / SP500_POOLED_RANK_IC):+.2f}"
        if not np.isnan(pooled['mean']) and SP500_POOLED_RANK_IC != 0
        else "TBD"
    )
    ratio_nasdaq = (
        f"{(pooled['mean'] / NASDAQ100_POOLED_RANK_IC):+.2f}"
        if not np.isnan(pooled['mean']) and NASDAQ100_POOLED_RANK_IC != 0
        else "TBD"
    )
    ratio_djia = (
        f"{(pooled['mean'] / DJIA30_POOLED_RANK_IC):+.2f}"
        if not np.isnan(pooled['mean']) and DJIA30_POOLED_RANK_IC != 0
        else "TBD"
    )
    cmp_block = (
        "## Comparison vs S&P 500 (canonical), NASDAQ-100, DJIA-30\n\n"
        "| Universe | Pooled rank IC | Std |\n"
        "|----------|----------------|-----|\n"
        f"| sp500    | +{SP500_POOLED_RANK_IC:.4f} (canonical) | "
        f"{SP500_POOLED_STD:.3f} |\n"
        f"| nasdaq100 | +{NASDAQ100_POOLED_RANK_IC:.4f} (parked) | n/a |\n"
        f"| djia30   | +{DJIA30_POOLED_RANK_IC:.4f} (parked) | n/a |\n"
        f"| biotech_nbi | {pooled['mean']:+.4f} | {pooled['std']:.4f} |\n"
        f"| ratio (biotech_nbi / sp500) | {ratio_sp500} | n/a |\n"
        f"| ratio (biotech_nbi / nasdaq100) | {ratio_nasdaq} | n/a |\n"
        f"| ratio (biotech_nbi / djia30) | {ratio_djia} | n/a |\n"
    )
    text = re.sub(
        r"## Comparison vs S&P 500.*?(?=\n## )",
        cmp_block + "\n",
        text,
        flags=re.DOTALL,
    )

    # Acceptance criteria.
    accept_block = (
        "## Acceptance criteria check\n\n"
        f"- F2 rate-stress non-negative: {acceptance['f2_non_negative']}\n"
        f"- Pooled within +-30% of S&P 500: "
        f"{acceptance['pooled_within_30pct_sp500']}\n"
        f"- No NaN or training divergence: {acceptance['nan_free']}\n"
    )
    text = re.sub(
        r"## Acceptance criteria check.*?(?=\n## )",
        accept_block + "\n",
        text,
        flags=re.DOTALL,
    )

    report_path.write_text(text)
    print(f"[rollup] patched {report_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-dir", type=Path, default=METRICS_DIR_DEFAULT)
    p.add_argument("--report", type=Path, default=REPORT_PATH_DEFAULT)
    p.add_argument("--no-patch", action="store_true",
                   help="Print only; do not patch the report.")
    args = p.parse_args()

    cells = _read_cells(args.metrics_dir)
    if not cells:
        print(f"[rollup] no cell metrics found under {args.metrics_dir}")
        return 1
    print(f"[rollup] loaded {len(cells)} cell metrics from {args.metrics_dir}")

    # Cohort sanity: val-rank-ic vs test-rank-ic per cell.
    val_vs_test = []
    for c in cells:
        vi = c.get("val_rank_ic", float("nan"))
        ti = c.get("test_rank_ic", float("nan"))
        val_vs_test.append((c["fold"], c["seed"], vi, ti))
    print("\nPer-cell val vs test rank IC (sanity-check val-based selection):")
    print("  fold seed  val_rank_ic  test_rank_ic")
    for f, s, vi, ti in sorted(val_vs_test):
        print(f"    F{f}  {s}    {vi:+.4f}      {ti:+.4f}")

    per_fold = _per_fold_rollup(cells)
    pooled = _pooled(cells)
    print("\nPer-fold pooled rank IC (mean +- std, n):")
    for f in [1, 2, 3, 4, 5]:
        d = per_fold[f]
        print(
            f"  F{f}: {d['mean']:+.4f} +- {d['std']:.4f}  "
            f"(n={d['n']})"
        )
    print(
        f"\nPooled across {pooled['n']} cells: "
        f"{pooled['mean']:+.4f} +- {pooled['std']:.4f}"
    )
    if not np.isnan(pooled["mean"]) and SP500_POOLED_RANK_IC != 0:
        ratio = pooled["mean"] / SP500_POOLED_RANK_IC
        print(
            f"vs sp500 canonical +{SP500_POOLED_RANK_IC:.4f}: "
            f"ratio = {ratio:+.2f}"
        )
    if not np.isnan(pooled["mean"]) and NASDAQ100_POOLED_RANK_IC != 0:
        ratio_n = pooled["mean"] / NASDAQ100_POOLED_RANK_IC
        print(
            f"vs nasdaq100 parked +{NASDAQ100_POOLED_RANK_IC:.4f}: "
            f"ratio = {ratio_n:+.2f}"
        )
    if not np.isnan(pooled["mean"]) and DJIA30_POOLED_RANK_IC != 0:
        ratio_d = pooled["mean"] / DJIA30_POOLED_RANK_IC
        print(
            f"vs djia30 parked +{DJIA30_POOLED_RANK_IC:.4f}: "
            f"ratio = {ratio_d:+.2f}"
        )

    acceptance = _check_acceptance(per_fold, pooled, cells)
    print("\nAcceptance criteria:")
    for k, v in acceptance.items():
        print(f"  {k}: {v}")

    if not args.no_patch:
        _patch_report(args.report, per_fold, pooled, acceptance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
