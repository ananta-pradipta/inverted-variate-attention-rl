"""C3 sector-aware NBI sub-industry rollup: NBI 25-cell.

Reads NBI L2L3 daily-tape parquets (per cell SAC L/S strategy
return tapes) for C3 sector-aware pretrain (sub-industry granularity)
and the corresponding canonical NBI baseline tapes, then computes
per-cell annualised Sharpe, per-fold mean Sharpe, pool (25-cell mean),
delta vs canonical pool, and per-fold SoS.

This is a separate rollup from ``scripts/rollup_c3_sector_ndx_nbi.py``
which SKIPPED NBI for top-level sector degeneracy. The new C3 NBI run
uses SUB-INDUSTRY granularity (~73% Biotechnology + 6 minority
cohorts) so the InfoNCE term is non-degenerate.

Universe wiring (matches the L2L3 sbatch output roots):

  NBI canonical    : outputs/biotech_nbi_enriched/layer3_k25/ls/
                     fold*_seed*.parquet (equal_topk K=25 wrapper;
                     pool +1.541 reference)
  NBI C3 sector    : outputs/biotech_nbi_enriched/layer3_c3_sector_k25/
                     ls/fold*_seed*.parquet

Usage::

    python scripts/rollup_c3_sector_nbi_sub_industry.py \\
        --out reports/pretrain_improvements/c3_nbi_sub_industry_2026-05-27.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


TRADING_DAYS = 252

# Locked canonical reference for the stop gates (matches the NBI
# canonical pool reported in scripts/rollup_b1_hmm_ndx_nbi.py and
# scripts/rollup_c3_sector_ndx_nbi.py).
CANONICAL_NBI_POOL = 1.541
B1_HMM_NBI_POOL = 1.107
DSL_NBI_POOL = 1.199

# Stop gates per the task spec:
#   pool >= canonical -0.05 = +1.491 -> WITHIN FLOOR (no-lift)
#   pool >= canonical +0.05 = +1.591 -> WIN
#   pool >= canonical +0.10 = +1.641 -> STRONG WIN
NBI_FLOOR = CANONICAL_NBI_POOL - 0.050
NBI_WIN = CANONICAL_NBI_POOL + 0.050
NBI_STRONG_WIN = CANONICAL_NBI_POOL + 0.100


def _annualised_sharpe(rets: np.ndarray) -> float:
    """Mean / std * sqrt(252). Returns 0.0 on degenerate input."""
    if rets.size < 2:
        return 0.0
    sd = float(rets.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(rets.mean() / sd * math.sqrt(TRADING_DAYS))


def _parse_cell_key(name: str) -> Tuple[int, int] | None:
    """Extract (fold, seed) from a 'fold{F}_seed{S}' filename stem."""
    m = re.match(r"fold(\d+)_seed(\d+)", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _load_cell_sharpes(pattern: str) -> Dict[Tuple[int, int], float]:
    """Walk a glob of daily-tape parquets and produce (fold,seed)->Sharpe.

    Each parquet is expected to have a 'strategy_return' column with one
    row per test trading day. Cells whose parquet is missing the column
    or has < 2 rows are skipped (Sharpe undefined).
    """
    import pandas as pd
    out: Dict[Tuple[int, int], float] = {}
    for fp in sorted(glob.glob(pattern)):
        key = _parse_cell_key(Path(fp).stem)
        if key is None:
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if "strategy_return" not in df.columns:
            continue
        rets = df["strategy_return"].to_numpy(dtype=np.float64)
        s = _annualised_sharpe(rets)
        out[key] = s
    return out


def _per_fold_means(
    cells: Dict[Tuple[int, int], float],
) -> Dict[int, Tuple[float, int]]:
    """fold -> (mean Sharpe across seeds, n_cells)."""
    by_fold: Dict[int, List[float]] = {}
    for (f, _s), v in cells.items():
        by_fold.setdefault(int(f), []).append(float(v))
    return {f: (float(np.mean(vs)), len(vs)) for f, vs in by_fold.items()}


def _pool(cells: Dict[Tuple[int, int], float]) -> Tuple[float, int]:
    vs = list(cells.values())
    if not vs:
        return float("nan"), 0
    return float(np.mean(vs)), len(vs)


def _sos_per_fold(
    proposed: Dict[Tuple[int, int], float],
    baseline: Dict[Tuple[int, int], float],
) -> float:
    """Sign-of-Sign agreement: fraction of cells where the proposed
    has the same sign as the baseline (using only cells present in both).
    """
    keys = sorted(set(proposed) & set(baseline))
    if not keys:
        return float("nan")
    agree = 0
    for k in keys:
        sa = 1 if proposed[k] > 0 else (-1 if proposed[k] < 0 else 0)
        sb = 1 if baseline[k] > 0 else (-1 if baseline[k] < 0 else 0)
        if sa == sb and sa != 0:
            agree += 1
    return float(agree) / float(len(keys))


def _verdict(pool: float) -> str:
    if math.isnan(pool):
        return "INCOMPLETE (no cells)"
    if pool >= NBI_STRONG_WIN:
        return f"STRONG WIN (>= +{NBI_STRONG_WIN:.3f})"
    if pool >= NBI_WIN:
        return f"WIN (>= +{NBI_WIN:.3f})"
    if pool >= NBI_FLOOR:
        return (
            f"WITHIN FLOOR (>= +{NBI_FLOOR:.3f}, < +{NBI_WIN:.3f})"
        )
    return f"FAIL (< +{NBI_FLOOR:.3f})"


def _f4_f5_check(
    by_fold_prop: Dict[int, Tuple[float, int]],
    by_fold_base: Dict[int, Tuple[float, int]],
) -> Tuple[float, float, bool]:
    """Return (f4_f5_prop_sum, f4_f5_base_sum, pass_flag).

    Stop gate: F4 + F5 prop sum >= canonical -0.56 (don't worsen the
    already-weak folds). Canonical F4=+0.875, F5=+0.840 -> base sum
    +1.715; threshold = 1.715 - 0.56 = +1.155.
    """
    p4, _ = by_fold_prop.get(4, (float("nan"), 0))
    p5, _ = by_fold_prop.get(5, (float("nan"), 0))
    b4, _ = by_fold_base.get(4, (float("nan"), 0))
    b5, _ = by_fold_base.get(5, (float("nan"), 0))
    prop_sum = (
        (p4 + p5) if (not math.isnan(p4) and not math.isnan(p5))
        else float("nan")
    )
    base_sum = (
        (b4 + b5) if (not math.isnan(b4) and not math.isnan(b5))
        else float("nan")
    )
    threshold = base_sum - 0.56 if not math.isnan(base_sum) else float("nan")
    if math.isnan(prop_sum) or math.isnan(threshold):
        return prop_sum, base_sum, False
    return prop_sum, base_sum, prop_sum >= threshold


def _emit_universe_panel(
    proposed_glob: str,
    baseline_glob: str,
) -> Tuple[List[str], Dict[str, float], str]:
    """Build the NBI markdown panel and return (lines, stats, verdict)."""
    prop = _load_cell_sharpes(proposed_glob)
    base = _load_cell_sharpes(baseline_glob)
    by_fold_prop = _per_fold_means(prop)
    by_fold_base = _per_fold_means(base)
    pool_prop, n_prop = _pool(prop)
    pool_base, n_base = _pool(base)
    pool_delta = (
        (pool_prop - pool_base)
        if (not math.isnan(pool_prop) and not math.isnan(pool_base))
        else float("nan")
    )
    sos = _sos_per_fold(prop, base)
    verdict = _verdict(pool_prop)
    f45_prop_sum, f45_base_sum, f45_pass = _f4_f5_check(
        by_fold_prop, by_fold_base
    )

    lines: List[str] = []
    lines.append(
        "## biotech NBI-enriched "
        "(25-cell, C3 sector sub-industry Layer-1 -> SAC L/S K=25)"
    )
    lines.append("")
    lines.append(
        f"Canonical pool reference: **+{CANONICAL_NBI_POOL:.3f}**"
    )
    lines.append(
        f"Stop gates: floor +{NBI_FLOOR:.3f}; "
        f"WIN +{NBI_WIN:.3f}; STRONG WIN +{NBI_STRONG_WIN:.3f}"
    )
    lines.append(
        f"Comparators: B1 HMM NBI +{B1_HMM_NBI_POOL:.3f}; "
        f"DSL NBI +{DSL_NBI_POOL:.3f}"
    )
    lines.append("")
    lines.append("| Fold | canonical | C3 sub-industry | delta |")
    lines.append("|---:|---:|---:|---:|")
    folds = sorted(set(by_fold_prop) | set(by_fold_base))
    for f in folds:
        bp, _ = by_fold_prop.get(f, (float("nan"), 0))
        bb, _ = by_fold_base.get(f, (float("nan"), 0))
        d = (
            (bp - bb)
            if (not math.isnan(bp) and not math.isnan(bb))
            else float("nan")
        )
        lines.append(f"| F{f} | {bb:+.3f} | {bp:+.3f} | {d:+.3f} |")
    lines.append(
        f"| **Pool** | **{pool_base:+.3f}** | **{pool_prop:+.3f}** | "
        f"**{pool_delta:+.3f}** |"
    )
    lines.append(f"| SoS | -- | {sos:.3f} | -- |")
    lines.append("")
    lines.append(
        f"Cells: proposed={n_prop} baseline={n_base} (target 25 each)."
    )
    if not math.isnan(f45_prop_sum) and not math.isnan(f45_base_sum):
        f45_status = "PASS" if f45_pass else "FAIL"
        lines.append(
            f"F4+F5 check: prop sum {f45_prop_sum:+.3f} vs base sum "
            f"{f45_base_sum:+.3f}; threshold {f45_base_sum - 0.56:+.3f} "
            f"(base -0.56); {f45_status}"
        )
    lines.append(f"**Verdict (NBI sub-industry)**: {verdict}")
    lines.append("")
    stats = {
        "pool_prop": pool_prop,
        "pool_base": pool_base,
        "pool_delta": pool_delta,
        "n_prop": float(n_prop),
        "n_base": float(n_base),
        "sos": sos,
        "f45_prop_sum": f45_prop_sum,
        "f45_base_sum": f45_base_sum,
        "f45_pass": float(f45_pass),
    }
    return lines, stats, verdict


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--nbi-proposed-glob",
        type=str,
        default=(
            "outputs/biotech_nbi_enriched/layer3_c3_sector_k25/ls/"
            "fold*_seed*.parquet"
        ),
    )
    p.add_argument(
        "--nbi-baseline-glob",
        type=str,
        default=(
            "outputs/biotech_nbi_enriched/layer3_k25/ls/"
            "fold*_seed*.parquet"
        ),
    )
    p.add_argument(
        "--out",
        type=str,
        default=(
            "reports/pretrain_improvements/"
            "c3_nbi_sub_industry_2026-05-27.md"
        ),
    )
    p.add_argument(
        "--json-out",
        type=str,
        default=(
            "reports/pretrain_improvements/"
            "c3_nbi_sub_industry_2026-05-27.json"
        ),
    )
    args = p.parse_args()

    lines: List[str] = []
    lines.append(
        "# Canonical InVAR-RL C3 (Sector-Aware Positives Pretrain): "
        "NBI sub-industry"
    )
    lines.append("")
    lines.append(
        "Updated 2026-05-27. C3 cross-universe transfer test on the "
        "biotech NBI-enriched panel using SUB-INDUSTRY granularity "
        "(not the top-level GICS 'Health Care' cohort that originally "
        "caused the C3 NBI rollup to SKIP). 9 healthcare-focused "
        "sub-industry cohorts compiled hand on 2026-05-27 (Biotechnology "
        "default + 6 explicit minority cohorts in the populated cache). "
        "Same C3 hook as the SP500 winner (pool +1.046 vs canonical "
        "+0.945, +0.101 lift); only the panel kind and the sub-industry "
        "cache differ."
    )
    lines.append("")

    panel_lines, stats, verdict = _emit_universe_panel(
        proposed_glob=args.nbi_proposed_glob,
        baseline_glob=args.nbi_baseline_glob,
    )
    lines.extend(panel_lines)

    # Overall verdict.
    lines.append("## Overall verdict")
    lines.append("")
    pp = stats["pool_prop"]
    if not math.isnan(pp):
        if pp >= NBI_STRONG_WIN:
            overall = (
                f"NBI sub-industry STRONG WIN at +{pp:.3f} vs canonical "
                f"+{CANONICAL_NBI_POOL:.3f} (delta {stats['pool_delta']:+.3f}). "
                "C3 with sub-industry granularity transfers the SP500 win "
                "to the biotech universe. Recommended next step: re-pitch "
                "C3 as a cross-universe candidate on the three universes "
                "(SP500 + NDX + NBI sub-industry) and elevate the "
                "sub-industry mapping to a permanent NBI artefact."
            )
        elif pp >= NBI_WIN:
            overall = (
                f"NBI sub-industry WIN at +{pp:.3f} vs canonical "
                f"+{CANONICAL_NBI_POOL:.3f} (delta {stats['pool_delta']:+.3f}). "
                "C3 with sub-industry granularity transfers cross-universe."
            )
        elif pp >= NBI_FLOOR:
            overall = (
                f"NBI sub-industry WITHIN FLOOR at +{pp:.3f} "
                f"(delta {stats['pool_delta']:+.3f}). C3 sub-industry "
                "is non-regressive cross-universe but does not clear "
                "the WIN gate; not a new headline."
            )
        else:
            overall = (
                f"NBI sub-industry FAIL at +{pp:.3f} "
                f"(delta {stats['pool_delta']:+.3f}). The sub-industry "
                "rescue does not recover the SP500 lift on NBI; C3 is "
                "SP500/NDX-specific. Recommended next step: archive "
                "C3 NBI alongside the original sector-degenerate SKIP."
            )
    else:
        overall = "INCOMPLETE: no proposed cells loaded."

    lines.append(f"- NBI sub-industry: {verdict}")
    lines.append(f"- **{overall}**")
    lines.append("")

    # Reproducibility.
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        "- C3 code (READ-ONLY): "
        "`src/models/pretrain_improvements/sector_positives.py` + "
        "`src/baselines/train_invar_clpretrain_v2.py` (sector hook in "
        "Stage 1)."
    )
    lines.append(
        "- NBI Stage-1 entrypoint: "
        "`invar_rl/training/biotech_nbi_enriched_layer1_c3_sector.py`."
    )
    lines.append(
        "- Sbatches: "
        "`invar_rl/scripts/wulver/invar_rl_biotech_nbi_enriched_"
        "c3_sector_{stage1,l2l3}.sbatch`."
    )
    lines.append(
        "- NBI sub-industry cache: "
        "`cache/sector_labels/biotech_nbi_enriched.parquet` (built "
        "once via "
        "`invar_rl.scripts.build_biotech_nbi_enriched_sector_map`; "
        "351/351 = 100% coverage, 7 sub-industries populated)."
    )
    lines.append(
        "- Outputs: `outputs/biotech_nbi_enriched/layer1_c3_sector/`, "
        "`outputs/biotech_nbi_enriched/layer3_c3_sector_k25/`."
    )
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[INFO] wrote {out_path}")
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps({
        "nbi_sub_industry": stats,
        "verdict": verdict,
        "overall": overall,
        "canonical_nbi_pool": CANONICAL_NBI_POOL,
        "b1_hmm_nbi_pool": B1_HMM_NBI_POOL,
        "dsl_nbi_pool": DSL_NBI_POOL,
    }, indent=2))
    print(f"[INFO] wrote {args.json_out}")
    for ln in lines:
        print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
