"""Roll up SP500 Stage 1 (Layer 1 canonical InVAR) retrain results.

Reads every ``invar_rl/results/stage1/fold{F}_seed{S}.json`` written by
``invar_rl.training.stage1_rank``, computes per-fold pooled rank IC,
the all-25-cell pooled rank IC, and the regression check vs the
canonical InVAR locked-2026-05-19 reference (+0.0284 pooled rank IC).

Writes a markdown report to ``reports/sp500/stage1_retrain_2026-05-23.md``.

Usage::

    PYTHONPATH=$PWD python3 -m invar_rl.scripts.rollup_sp500_stage1_retrain
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


METRICS_DIR_DEFAULT = Path("invar_rl/results/stage1")
REPORT_PATH_DEFAULT = Path("reports/sp500/stage1_retrain_2026-05-23.md")

CANONICAL_POOLED_RANK_IC = 0.0284
CANONICAL_POOLED_STD = 0.019
DRIFT_THRESHOLD = 0.005  # |pooled - canonical| <= this is acceptable

FOLD_LABELS = {
    1: "F1",
    2: "F2",
    3: "F3",
    4: "F4",
    5: "F5",
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
    out: Dict[int, Dict[str, float]] = {}
    for f in [1, 2, 3, 4, 5]:
        vals = [
            c["rank_ic"] for c in cells
            if c.get("fold") == f and "rank_ic" in c
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
    vals = [c["rank_ic"] for c in cells if "rank_ic" in c]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def _render_markdown(
    cells: List[dict],
    per_fold: Dict[int, Dict[str, float]],
    pooled: Dict[str, float],
    out_path: Path,
) -> None:
    drift = pooled["mean"] - CANONICAL_POOLED_RANK_IC
    drift_ok = abs(drift) <= DRIFT_THRESHOLD if not np.isnan(pooled["mean"]) else False

    lines: List[str] = []
    lines.append("# SP500 Layer 1 (canonical InVAR) retrain on 2755-day panel")
    lines.append("")
    lines.append(
        "Path B Phase B3 rollup. Re-trained or re-validated the 25-cell "
        "(5 folds x 5 seeds) canonical InVAR sweep (bankless + macro-state "
        "contrastive InfoNCE clpretrain) against the rebuilt 2755-day "
        "lattice panel (`data/lattice/processed/panel_features.parquet`, "
        "rebuild commit Path B 2026-05-23)."
    )
    lines.append("")

    n_total = len(cells)
    if cells:
        panel_T_vals = [c.get("panel_T") for c in cells if "panel_T" in c]
        panel_N_vals = [c.get("panel_N") for c in cells if "panel_N" in c]
        if panel_T_vals:
            unique_T = sorted(set(panel_T_vals))
            unique_N = sorted(set(panel_N_vals)) if panel_N_vals else []
            lines.append(f"- panel_T (unique): {unique_T}")
            lines.append(f"- panel_N (unique): {unique_N}")
            lines.append(f"- total cells: {n_total}")
            lines.append("")

    lines.append("## Per-fold pooled rank IC (5 seeds per fold)")
    lines.append("")
    lines.append("| fold | mean | std | n |")
    lines.append("|------|------|-----|---|")
    for f in [1, 2, 3, 4, 5]:
        d = per_fold[f]
        if d["n"]:
            lines.append(
                f"| {FOLD_LABELS[f]} | {d['mean']:+.4f} | "
                f"{d['std']:.4f} | {d['n']} |"
            )
        else:
            lines.append(f"| {FOLD_LABELS[f]} | TBD | TBD | 0 |")
    lines.append("")

    lines.append(f"## Pooled across all {n_total} cells")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|--------|------:|")
    lines.append(f"| pooled rank IC mean | {pooled['mean']:+.4f} |")
    lines.append(f"| pooled rank IC std  | {pooled['std']:.4f} |")
    lines.append(f"| n_cells             | {pooled['n']} |")
    lines.append("")

    lines.append("## Regression check vs canonical (locked 2026-05-19)")
    lines.append("")
    lines.append("| metric | this run | canonical | delta |")
    lines.append("|--------|---------:|----------:|------:|")
    lines.append(
        f"| pooled rank IC | {pooled['mean']:+.4f} | "
        f"+{CANONICAL_POOLED_RANK_IC:.4f} | {drift:+.4f} |"
    )
    lines.append("")
    if drift_ok:
        lines.append(
            f"PASS: pooled rank IC drift |{drift:+.4f}| is within the "
            f"+/-{DRIFT_THRESHOLD:.3f} regression threshold. The retrained "
            "Layer 1 checkpoints reproduce the canonical InVAR reference "
            "on the rebuilt 2755-day panel."
        )
    else:
        lines.append(
            f"FAIL: pooled rank IC drift |{drift:+.4f}| exceeds the "
            f"+/-{DRIFT_THRESHOLD:.3f} regression threshold. Investigate "
            "feature scaling, fold cutoffs, or raw-input drift before "
            "running downstream stages."
        )
    lines.append("")

    lines.append("## Per-cell drill-down")
    lines.append("")
    lines.append("| fold | seed | panel_T | panel_N | rank_ic | val_rank_ic |")
    lines.append("|-----:|-----:|--------:|--------:|--------:|------------:|")
    for c in sorted(cells, key=lambda x: (x.get("fold", 0), x.get("seed", 0))):
        lines.append(
            f"| {c.get('fold'):>4} | {c.get('seed'):>4} | "
            f"{c.get('panel_T', 'n/a'):>7} | "
            f"{c.get('panel_N', 'n/a'):>7} | "
            f"{c.get('rank_ic', float('nan')):+.4f} | "
            f"{c.get('val_rank_ic', float('nan')):+.4f} |"
        )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> int:
    cells = _read_cells(METRICS_DIR_DEFAULT)
    if not cells:
        print(f"[rollup stage1] no cell metrics under {METRICS_DIR_DEFAULT}")
        return 1
    print(f"[rollup stage1] loaded {len(cells)} cells")

    per_fold = _per_fold_rollup(cells)
    pooled = _pooled(cells)

    print(f"\nPer-fold pooled rank IC:")
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
    drift = pooled["mean"] - CANONICAL_POOLED_RANK_IC
    print(
        f"Canonical reference: +{CANONICAL_POOLED_RANK_IC:.4f}; "
        f"drift={drift:+.4f}; threshold +/-{DRIFT_THRESHOLD:.3f}; "
        f"PASS" if abs(drift) <= DRIFT_THRESHOLD else "FAIL"
    )

    _render_markdown(cells, per_fold, pooled, REPORT_PATH_DEFAULT)
    print(f"\n[rollup stage1] wrote {REPORT_PATH_DEFAULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
