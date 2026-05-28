"""B1 HMM cross-universe rollup: NDX-100 + NBI-enriched 25-cell each.

Reads B1 HMM L2L3 daily-tape parquets (per cell SAC L/S strategy
return tapes) and the corresponding canonical baseline tapes, then
computes:
  - per-cell annualised Sharpe (mean / std * sqrt(252)),
  - per-fold mean Sharpe (n cells),
  - pool (25-cell mean),
  - delta vs canonical pool,
  - per-fold SoS (sign-of-sign agreement) summary.

Mirrors the SP500 B1 rollup style in
reports/pretrain_improvements/b1_hmm_2026-05-27.md.

Universe wiring (matches the L2L3 sbatch output roots):

  NDX-100 canonical    : outputs/nasdaq100/layer3/ls/fold*_seed*.parquet
                         (QP default wrapper; pool +1.194 reference)
  NDX-100 B1 HMM       : outputs/nasdaq100/layer3_b1_hmm/ls/fold*_seed*.parquet

  NBI-enriched canon   : outputs/biotech_nbi_enriched/layer3_k25/ls/
                         fold*_seed*.parquet
                         (equal_topk K=25 wrapper; pool +1.541 reference)
  NBI-enriched B1 HMM  : outputs/biotech_nbi_enriched/layer3_b1_hmm_k25/ls/
                         fold*_seed*.parquet

Usage::

    python scripts/rollup_b1_hmm_ndx_nbi.py \
        --out reports/pretrain_improvements/b1_hmm_ndx_nbi_2026-05-27.md
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

# Locked canonical references for the stop gates (from the prompt + the
# repo's existing canonical pools, verified by recomputation on the
# 25-cell canonical tapes during planning).
CANONICAL_NDX_POOL = 1.194
CANONICAL_NBI_POOL = 1.541

NDX_WIN = CANONICAL_NDX_POOL + 0.020
NDX_STRONG_WIN = CANONICAL_NDX_POOL + 0.050
NDX_FLOOR = CANONICAL_NDX_POOL - 0.050

NBI_WIN = CANONICAL_NBI_POOL + 0.020
NBI_STRONG_WIN = CANONICAL_NBI_POOL + 0.050
NBI_FLOOR = CANONICAL_NBI_POOL - 0.050


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


def _verdict(pool: float, win: float, strong: float, floor: float) -> str:
    if math.isnan(pool):
        return "INCOMPLETE (no cells)"
    if pool >= strong:
        return f"STRONG WIN (>= +{strong:.3f})"
    if pool >= win:
        return f"WIN (>= +{win:.3f})"
    if pool >= floor:
        return f"WITHIN FLOOR (>= +{floor:.3f}, < +{win:.3f})"
    return f"FAIL (< +{floor:.3f})"


def _emit_universe_panel(
    name: str,
    proposed_glob: str,
    baseline_glob: str,
    canonical_pool: float,
    win: float,
    strong: float,
    floor: float,
) -> Tuple[List[str], Dict[str, float], str]:
    """Build the per-universe markdown panel and return (lines, stats, verdict).

    stats keys: pool_prop, pool_base, pool_delta, n_prop, n_base, sos.
    """
    prop = _load_cell_sharpes(proposed_glob)
    base = _load_cell_sharpes(baseline_glob)
    by_fold_prop = _per_fold_means(prop)
    by_fold_base = _per_fold_means(base)
    pool_prop, n_prop = _pool(prop)
    pool_base, n_base = _pool(base)
    pool_delta = (pool_prop - pool_base) if (not math.isnan(pool_prop) and not math.isnan(pool_base)) else float("nan")
    sos = _sos_per_fold(prop, base)
    verdict = _verdict(pool_prop, win, strong, floor)

    lines: List[str] = []
    lines.append(f"## {name}")
    lines.append("")
    lines.append(f"Canonical pool reference: **+{canonical_pool:.3f}**")
    lines.append(
        f"Stop gates: floor +{floor:.3f}; WIN +{win:.3f}; STRONG WIN +{strong:.3f}"
    )
    lines.append("")
    lines.append(
        "| Fold | canonical | B1 HMM | delta |"
    )
    lines.append("|---:|---:|---:|---:|")
    folds = sorted(set(by_fold_prop) | set(by_fold_base))
    for f in folds:
        bp, _ = by_fold_prop.get(f, (float("nan"), 0))
        bb, _ = by_fold_base.get(f, (float("nan"), 0))
        d = (bp - bb) if (not math.isnan(bp) and not math.isnan(bb)) else float("nan")
        lines.append(
            f"| F{f} | {bb:+.3f} | {bp:+.3f} | {d:+.3f} |"
        )
    lines.append(
        f"| **Pool** | **{pool_base:+.3f}** | **{pool_prop:+.3f}** | **{pool_delta:+.3f}** |"
    )
    lines.append(f"| SoS | -- | {sos:.3f} | -- |")
    lines.append("")
    lines.append(
        f"Cells: proposed={n_prop} baseline={n_base} (target 25 each)."
    )
    lines.append(f"**Verdict ({name})**: {verdict}")
    lines.append("")
    stats = {
        "pool_prop": pool_prop,
        "pool_base": pool_base,
        "pool_delta": pool_delta,
        "n_prop": float(n_prop),
        "n_base": float(n_base),
        "sos": sos,
    }
    return lines, stats, verdict


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ndx-proposed-glob",
        type=str,
        default="outputs/nasdaq100/layer3_b1_hmm/ls/fold*_seed*.parquet",
    )
    p.add_argument(
        "--ndx-baseline-glob",
        type=str,
        default="outputs/nasdaq100/layer3/ls/fold*_seed*.parquet",
    )
    p.add_argument(
        "--nbi-proposed-glob",
        type=str,
        default="outputs/biotech_nbi_enriched/layer3_b1_hmm_k25/ls/fold*_seed*.parquet",
    )
    p.add_argument(
        "--nbi-baseline-glob",
        type=str,
        default="outputs/biotech_nbi_enriched/layer3_k25/ls/fold*_seed*.parquet",
    )
    p.add_argument(
        "--out",
        type=str,
        default="reports/pretrain_improvements/b1_hmm_ndx_nbi_2026-05-27.md",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default="reports/pretrain_improvements/b1_hmm_ndx_nbi_2026-05-27.json",
    )
    args = p.parse_args()

    lines: List[str] = []
    lines.append("# Canonical InVAR-RL B1 (HMM Regime Pretrain): NDX + NBI cross-universe")
    lines.append("")
    lines.append(
        "Updated 2026-05-27. B1 cross-universe transfer test on NASDAQ-100 "
        "and biotech NBI-enriched. Same B1 hook as the SP500 winner "
        "(pool +0.979 vs canonical +0.945, +0.034 lift); only the panel "
        "kind and per-universe SAC wrapper differ. Per-universe canonical "
        "references are: NDX +1.194 (layer3 QP default), NBI +1.541 "
        "(layer3_k25 equal_topk K=25)."
    )
    lines.append("")

    ndx_lines, ndx_stats, ndx_verdict = _emit_universe_panel(
        name="NASDAQ-100 (25-cell, B1 HMM Layer-1 -> SAC L/S K=QP)",
        proposed_glob=args.ndx_proposed_glob,
        baseline_glob=args.ndx_baseline_glob,
        canonical_pool=CANONICAL_NDX_POOL,
        win=NDX_WIN,
        strong=NDX_STRONG_WIN,
        floor=NDX_FLOOR,
    )
    nbi_lines, nbi_stats, nbi_verdict = _emit_universe_panel(
        name="biotech NBI-enriched (25-cell, B1 HMM Layer-1 -> SAC L/S K=25)",
        proposed_glob=args.nbi_proposed_glob,
        baseline_glob=args.nbi_baseline_glob,
        canonical_pool=CANONICAL_NBI_POOL,
        win=NBI_WIN,
        strong=NBI_STRONG_WIN,
        floor=NBI_FLOOR,
    )
    lines.extend(ndx_lines)
    lines.extend(nbi_lines)

    # Overall verdict.
    lines.append("## Overall verdict")
    lines.append("")
    ndx_win = ndx_stats["pool_prop"] >= NDX_WIN if not math.isnan(ndx_stats["pool_prop"]) else False
    nbi_win = nbi_stats["pool_prop"] >= NBI_WIN if not math.isnan(nbi_stats["pool_prop"]) else False
    if ndx_win and nbi_win:
        overall = "BOTH WIN: cross-universe B1 confirmed (paper-headline candidate)."
    elif ndx_win or nbi_win:
        overall = "ONE WIN: partial generalisation (still publishable)."
    else:
        overall = "BOTH FAIL: SP500-specific like DSL."
    lines.append(f"- NDX-100: {ndx_verdict}")
    lines.append(f"- NBI-enriched: {nbi_verdict}")
    lines.append(f"- **Combined: {overall}**")
    lines.append("")

    # Reproducibility.
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("- B1 code (READ-ONLY): `src/models/pretrain_improvements/hmm_regime.py` + `src/baselines/train_invar_clpretrain_v2.py` (HMM hook in Stage 1).")
    lines.append("- Per-universe Stage-1 entrypoints: `invar_rl/training/{nasdaq100,biotech_nbi_enriched}_layer1_b1_hmm.py`.")
    lines.append("- Sbatches: `invar_rl/scripts/wulver/invar_rl_{nasdaq100,biotech_nbi_enriched}_b1_hmm_{stage1,l2l3}.sbatch`.")
    lines.append("- HMM cache: `cache/pretrain_improvements/hmm_regime/{nasdaq100,biotech_nbi_enriched}/foldF/posteriors.parquet`.")
    lines.append("- Outputs: `outputs/nasdaq100/layer1_b1_hmm/`, `outputs/nasdaq100/layer3_b1_hmm/`, `outputs/biotech_nbi_enriched/layer1_b1_hmm/`, `outputs/biotech_nbi_enriched/layer3_b1_hmm_k25/`.")
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[INFO] wrote {out_path}")
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps({
        "ndx": ndx_stats,
        "nbi": nbi_stats,
        "ndx_verdict": ndx_verdict,
        "nbi_verdict": nbi_verdict,
        "overall": overall,
        "canonical_ndx_pool": CANONICAL_NDX_POOL,
        "canonical_nbi_pool": CANONICAL_NBI_POOL,
    }, indent=2))
    print(f"[INFO] wrote {args.json_out}")
    for ln in lines:
        print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
