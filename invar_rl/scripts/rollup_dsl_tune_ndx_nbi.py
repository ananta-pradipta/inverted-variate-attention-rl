"""Roll up the DSL tune-A (NDX) + tune-B (NBI) Phase 2 sweep.

Reads L2L3 summary JSONs under
``outputs/{universe}/stage3_rl_ablation/diff_sharpe_phase2_tune{A,B}/ls/summary/``
and emits per-universe stats compared to BOTH the canonical SAC L/S
reference AND the DSL default (tau=0.1, weight=0.2) reference.

References:
- NDX-100 canonical pool +1.194 ; DSL default pool +0.900
- NBI-enr  canonical pool +1.541 ; DSL default pool +1.199

Pre-registered tune-A verdict (NDX tau=0.05, weight=0.3):
- PASS  iff pool >= +1.144  (within -0.05 of canonical +1.194)
- IMPROVE  iff pool >= +0.900 (above DSL default)
- FAIL_HARD iff pool < +0.85

Pre-registered tune-B verdict (NBI weight=0.5, tau=0.1):
- PASS_POOL iff pool >= +1.199 (above DSL default)
- PASS_F45  iff F4+F5 >= +1.106 (above DSL default sum)
- FAIL  iff pool < +1.1
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


SEEDS = (42, 43, 44, 45, 46)
FOLDS = (1, 2, 3, 4, 5)

CANONICAL_POOL: Dict[str, float] = {
    "nasdaq100": +1.194,
    "biotech_nbi_enriched": +1.541,
}
CANONICAL_PER_FOLD: Dict[str, Dict[int, float]] = {
    "nasdaq100": {1: +1.97, 2: -0.34, 3: +0.72, 4: +1.74, 5: +1.87},
    "biotech_nbi_enriched": {
        1: +2.43, 2: +1.91, 3: +0.88, 4: -0.29, 5: -0.27,
    },
}
DSL_DEFAULT_POOL: Dict[str, float] = {
    "nasdaq100": +0.900,
    "biotech_nbi_enriched": +1.199,
}
DSL_DEFAULT_PER_FOLD: Dict[str, Dict[int, float]] = {
    "nasdaq100": {1: +1.689, 2: -0.463, 3: +0.149, 4: +0.925, 5: +2.203},
    "biotech_nbi_enriched": {
        1: +2.044, 2: +1.848, 3: +0.999, 4: +1.003, 5: +0.103,
    },
}

DEFAULT_TUNE_DIR: Dict[str, Path] = {
    "nasdaq100": Path(
        "outputs/nasdaq100/stage3_rl_ablation/diff_sharpe_phase2_tuneA/ls"
    ),
    "biotech_nbi_enriched": Path(
        "outputs/biotech_nbi_enriched/stage3_rl_ablation/"
        "diff_sharpe_phase2_tuneB/ls"
    ),
}

TUNE_LABEL: Dict[str, str] = {
    "nasdaq100": "tune A (tau=0.05, weight=0.3)",
    "biotech_nbi_enriched": "tune B (tau=0.1, weight=0.5)",
}


def _load_cell(root: Path, fold: int, seed: int) -> float:
    """Read test_pooled_sharpe from one cell's summary JSON; NaN if missing."""
    p = root / "summary" / f"fold{fold}_seed{seed}.json"
    if not p.exists():
        return float("nan")
    with open(p) as fh:
        d = json.load(fh)
    return float(d.get("test_pooled_sharpe", float("nan")))


def _mean_sd(xs: List[float]) -> Tuple[float, float]:
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def _sos(xs: List[float]) -> float:
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan")
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs)


def _per_universe_summary(universe: str, root: Path) -> Dict[str, object]:
    matrix: Dict[int, Dict[int, float]] = {}
    all_cells: List[float] = []
    per_fold_means: Dict[int, float] = {}
    per_fold_sds: Dict[int, float] = {}
    per_fold_cells: Dict[int, List[float]] = {}
    for fold in FOLDS:
        row: Dict[int, float] = {}
        for seed in SEEDS:
            v = _load_cell(root, fold, seed)
            row[seed] = v
            if v == v:
                all_cells.append(v)
        per_fold_cells[fold] = [row[s] for s in SEEDS]
        m, sd = _mean_sd(per_fold_cells[fold])
        per_fold_means[fold] = m
        per_fold_sds[fold] = sd
        matrix[fold] = row

    pool_mean, pool_sd = _mean_sd(all_cells)
    pool_sos = _sos(all_cells)

    return {
        "universe": universe,
        "n_cells": len(all_cells),
        "matrix": matrix,
        "per_fold_mean": per_fold_means,
        "per_fold_sd": per_fold_sds,
        "pool_mean": pool_mean,
        "pool_sd": pool_sd,
        "pool_sos": pool_sos,
    }


def _verdict_block(universe: str, s: Dict[str, object]) -> Dict[str, object]:
    pool = float(s["pool_mean"])
    canon = CANONICAL_POOL[universe]
    dsl_default = DSL_DEFAULT_POOL[universe]
    delta_canon = pool - canon if pool == pool else float("nan")
    delta_default = pool - dsl_default if pool == pool else float("nan")

    if universe == "nasdaq100":
        if pool != pool:
            verdict = "NO_DATA"
        elif pool < 0.85:
            verdict = "FAIL_HARD"
        elif pool >= (canon - 0.05):
            verdict = "PASS"
        elif pool >= dsl_default:
            verdict = "IMPROVE"
        else:
            verdict = "FAIL"
        return {
            "verdict": verdict,
            "pool": pool,
            "canonical": canon,
            "dsl_default": dsl_default,
            "delta_vs_canonical": delta_canon,
            "delta_vs_dsl_default": delta_default,
        }

    if universe == "biotech_nbi_enriched":
        pf = s["per_fold_mean"]
        f45 = float(pf[4]) + float(pf[5])
        f45_default = (
            DSL_DEFAULT_PER_FOLD[universe][4]
            + DSL_DEFAULT_PER_FOLD[universe][5]
        )
        pass_pool = pool == pool and pool >= dsl_default
        pass_f45 = f45 == f45 and f45 >= f45_default
        if pool != pool:
            verdict = "NO_DATA"
        elif pool < 1.1:
            verdict = "FAIL"
        elif pass_pool and pass_f45:
            verdict = "PASS_BOTH"
        elif pass_pool:
            verdict = "PASS_POOL_ONLY"
        elif pass_f45:
            verdict = "PASS_F45_ONLY"
        else:
            verdict = "FAIL"
        return {
            "verdict": verdict,
            "pool": pool,
            "canonical": canon,
            "dsl_default": dsl_default,
            "delta_vs_canonical": delta_canon,
            "delta_vs_dsl_default": delta_default,
            "f4_plus_f5": f45,
            "f4_plus_f5_default": f45_default,
            "f4_plus_f5_delta_vs_default": f45 - f45_default,
        }

    raise ValueError(f"unknown universe={universe!r}")


def _format_per_fold_table(
    universe: str, s: Dict[str, object],
) -> str:
    pf_mean = s["per_fold_mean"]
    pf_sd = s["per_fold_sd"]
    canonical_pf = CANONICAL_PER_FOLD[universe]
    default_pf = DSL_DEFAULT_PER_FOLD[universe]
    lines = []
    lines.append(
        f"### Per-fold ({universe}) {TUNE_LABEL[universe]} "
        f"vs canonical SAC L/S and DSL default\n"
    )
    lines.append(
        "| fold | canonical | DSL default | tune | tune sd | "
        "delta vs canonical | delta vs DSL default |"
    )
    lines.append(
        "|------|----------:|------------:|-----:|--------:|"
        "-------------------:|---------------------:|"
    )
    for f in FOLDS:
        m = float(pf_mean[f])
        sd = float(pf_sd[f])
        canon = canonical_pf[f]
        default = default_pf[f]
        d_canon = m - canon if m == m else float("nan")
        d_default = m - default if m == m else float("nan")
        m_s = f"{m:+.4f}" if m == m else "  NaN "
        sd_s = f"{sd:.4f}" if sd == sd else "  NaN "
        dc_s = f"{d_canon:+.4f}" if d_canon == d_canon else "  NaN "
        dd_s = f"{d_default:+.4f}" if d_default == d_default else "  NaN "
        lines.append(
            f"| F{f} | {canon:+.4f} | {default:+.4f} | {m_s} | {sd_s} | "
            f"{dc_s} | {dd_s} |"
        )
    return "\n".join(lines)


def _format_pool_table(summaries: Dict[str, Dict[str, object]]) -> str:
    lines = []
    lines.append("## Cross-universe pool summary (tune)\n")
    lines.append(
        "| universe              | canonical | DSL default | tune pool | "
        "tune sd  | delta vs canonical | delta vs DSL default | verdict |"
    )
    lines.append(
        "|-----------------------|----------:|------------:|----------:|"
        "---------:|-------------------:|---------------------:|--------:|"
    )
    for u, s in summaries.items():
        vb = s["verdict_block"]
        pool = vb["pool"]
        canon = vb["canonical"]
        default = vb["dsl_default"]
        sd = float(s["pool_sd"])
        d_canon = vb["delta_vs_canonical"]
        d_default = vb["delta_vs_dsl_default"]
        v = vb["verdict"]
        pool_s = f"{pool:+.4f}" if pool == pool else "  NaN "
        sd_s = f"{sd:.4f}" if sd == sd else "  NaN "
        dc_s = f"{d_canon:+.4f}" if d_canon == d_canon else "  NaN "
        dd_s = f"{d_default:+.4f}" if d_default == d_default else "  NaN "
        lines.append(
            f"| {u:<21} | {canon:+.4f} | {default:+.4f} | {pool_s} | "
            f"{sd_s} | {dc_s} | {dd_s} | {v} |"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Roll up the DSL tune-A (NDX tau=0.05 w=0.3) and tune-B "
            "(NBI w=0.5) Phase 2 sweeps. Compares against BOTH canonical "
            "SAC L/S reference and the DSL default reference."
        )
    )
    p.add_argument(
        "--out", type=str,
        default="reports/generalizable/dsl_tune_ndx_nbi_2026-05-27.md",
    )
    p.add_argument(
        "--nasdaq100-dir", type=str,
        default=str(DEFAULT_TUNE_DIR["nasdaq100"]),
    )
    p.add_argument(
        "--biotech-nbi-enriched-dir", type=str,
        default=str(DEFAULT_TUNE_DIR["biotech_nbi_enriched"]),
    )
    args = p.parse_args()

    universe_dirs = {
        "nasdaq100": Path(args.nasdaq100_dir),
        "biotech_nbi_enriched": Path(args.biotech_nbi_enriched_dir),
    }

    summaries: Dict[str, Dict[str, object]] = {}
    for u, root in universe_dirs.items():
        s = _per_universe_summary(u, root)
        s["verdict_block"] = _verdict_block(u, s)
        summaries[u] = s

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(
        "# DSL Phase 2 tune A (NDX) + tune B (NBI) sweep summary\n"
    )
    lines.append(
        "Tune-A NDX: tau=0.05, weight=0.3, K=20 (sharper soft top-K + "
        "higher Sharpe-surrogate weight). Tune-B NBI: weight=0.5, "
        "tau=0.1 unchanged, K=25 (amplify weak-fold recovery).\n"
    )
    lines.append(
        "DSL default reference (commit 7be80b3 cross-universe run): "
        "NDX pool +0.900, NBI pool +1.199.\n"
    )
    lines.append(_format_pool_table(summaries))
    lines.append("")
    for u in ("nasdaq100", "biotech_nbi_enriched"):
        lines.append(_format_per_fold_table(u, summaries[u]))
        lines.append("")

    lines.append("## Per-universe verdict criteria (pre-registered)\n")
    lines.append("Tune-A NDX (tau=0.05, weight=0.3):")
    lines.append("- PASS  iff pool >= +1.144 (within -0.05 of canonical)")
    lines.append("- IMPROVE iff pool >= +0.900 (above DSL default)")
    lines.append("- FAIL_HARD iff pool < +0.85")
    lines.append("")
    lines.append("Tune-B NBI (weight=0.5, tau=0.1):")
    lines.append("- PASS_BOTH      iff pool >= +1.199 AND F4+F5 >= +1.106")
    lines.append("- PASS_POOL_ONLY iff pool >= +1.199 only")
    lines.append("- PASS_F45_ONLY  iff F4+F5 >= +1.106 only")
    lines.append("- FAIL iff pool < +1.1")
    lines.append("")

    lines.append("## NBI tune-B F4+F5 detail\n")
    nb = summaries["biotech_nbi_enriched"]["verdict_block"]
    if "f4_plus_f5" in nb:
        f45 = float(nb["f4_plus_f5"])
        f45_def = float(nb["f4_plus_f5_default"])
        d = float(nb["f4_plus_f5_delta_vs_default"])
        f45_s = f"{f45:+.4f}" if f45 == f45 else "  NaN "
        f45_def_s = f"{f45_def:+.4f}"
        d_s = f"{d:+.4f}" if d == d else "  NaN "
        lines.append(
            f"- tune-B F4+F5 sum: {f45_s}\n"
            f"- DSL default F4+F5 sum: {f45_def_s}\n"
            f"- delta vs DSL default: {d_s}\n"
        )

    lines.append("## Raw per-cell matrix\n")
    for u, s in summaries.items():
        lines.append(f"### {u} ({TUNE_LABEL[u]})\n")
        lines.append("| fold \\ seed | " + " | ".join(
            f"seed {s_}" for s_ in SEEDS
        ) + " |")
        lines.append("|---" + ("|---" * len(SEEDS)) + "|")
        for f in FOLDS:
            row = s["matrix"][f]
            cells = []
            for sd in SEEDS:
                v = row[sd]
                cells.append(f"{v:+.4f}" if v == v else "  NaN ")
            lines.append(f"| F{f} | " + " | ".join(cells) + " |")
        lines.append("")

    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as fh:
        fh.write(text)
    print(f"[rollup_dsl_tune_ndx_nbi] wrote {out_path}")
    for u, s in summaries.items():
        vb = s["verdict_block"]
        pool = vb["pool"]
        canon = vb["canonical"]
        default = vb["dsl_default"]
        dc = vb["delta_vs_canonical"]
        dd = vb["delta_vs_dsl_default"]
        print(
            f"  {u} ({TUNE_LABEL[u]}): pool={pool:+.4f} "
            f"(canonical {canon:+.4f}, default {default:+.4f}, "
            f"delta_canon {dc:+.4f}, delta_default {dd:+.4f}) "
            f"verdict={vb['verdict']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
