"""Roll up the DSL Phase 2 cross-universe sweep into per-universe summaries.

Reads the L2L3 summary JSONs under
``outputs/{universe}/stage3_rl_ablation/diff_sharpe_phase2/ls/summary/``
and emits per-universe stats:
- per-fold mean + sd over 5 seeds
- pool 25-cell mean + sd + sum-of-squares (SoS)
- delta vs canonical SAC L/S pool reference
- per-fold delta vs canonical per-fold reference
- per-fold bootstrap p-value (paired, fold-as-unit)
- verdict per universe vs pre-registered success bands

References (canonical SAC L/S, pre-registered in the task spec):
- NDX-100   pool +1.194 (F1 +1.97, F2 -0.34, F3 +0.72, F4 +1.74, F5 +1.87)
- NBI-enr.  pool +1.541 (F1 +2.43, F2 +1.91, F3 +0.88, F4 -0.29, F5 -0.27)

Pre-registered success per universe:
- NDX-100: pool >= canonical - 0.05 AND F2 >= canonical F2 - 0.20
- NBI-enr: pool >= canonical - 0.05 AND F4+F5 >= canonical F4+F5 - 0.20

If DSL beats canonical pool on BOTH universes: STRONG cross-universe.
If DSL beats canonical pool on ONE:               PARTIAL (still publishable).
If DSL fails BOTH:                                SP500-conditional (negative).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


SEEDS = (42, 43, 44, 45, 46)
FOLDS = (1, 2, 3, 4, 5)

# Canonical SAC L/S references (matched to existing outputs at
# outputs/nasdaq100/layer3/ls/summary/ and
# outputs/biotech_nbi_enriched/layer3_k25/ls/summary/).
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

DEFAULT_DSL_DIR: Dict[str, Path] = {
    "nasdaq100": Path(
        "outputs/nasdaq100/stage3_rl_ablation/diff_sharpe_phase2/ls"
    ),
    "biotech_nbi_enriched": Path(
        "outputs/biotech_nbi_enriched/stage3_rl_ablation/"
        "diff_sharpe_phase2/ls"
    ),
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
    """Sample mean + ddof=1 sample std; (NaN, NaN) if no finite cells."""
    xs = [x for x in xs if x == x]  # drop NaNs
    if not xs:
        return float("nan"), float("nan")
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def _sos(xs: List[float]) -> float:
    """Sum of squares (centered); proxy for variance, scale-free."""
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan")
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs)


def _bootstrap_p_one_sided(
    dsl_cells: List[float],
    canonical_value: float,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> float:
    """One-sided bootstrap p-value: P(resample mean <= canonical | H0).

    H0: DSL mean = canonical_value (no transfer).
    Resample the DSL cells with replacement, recenter to the canonical
    value, and count what fraction of resamples have mean >= the
    observed DSL mean. Returns NaN if no finite DSL cells.
    """
    import random
    xs = [x for x in dsl_cells if x == x]
    if not xs:
        return float("nan")
    rng = random.Random(seed)
    obs = sum(xs) / len(xs)
    delta_obs = obs - canonical_value
    # Recenter cells to the null hypothesis mean (canonical_value).
    centered = [x - obs + canonical_value for x in xs]
    n = len(centered)
    hits = 0
    for _ in range(n_resamples):
        sample = [centered[rng.randrange(n)] for _ in range(n)]
        m = sum(sample) / n
        if (m - canonical_value) >= delta_obs:
            hits += 1
    return hits / n_resamples


def _per_universe_summary(
    universe: str, root: Path, n_bootstrap: int = 10_000,
) -> Dict[str, object]:
    """Compute the full per-universe DSL summary block."""
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

    canonical_pool = CANONICAL_POOL[universe]
    canonical_pf = CANONICAL_PER_FOLD[universe]

    pool_delta = pool_mean - canonical_pool if pool_mean == pool_mean else float("nan")
    per_fold_delta = {
        f: per_fold_means[f] - canonical_pf[f]
        if per_fold_means[f] == per_fold_means[f]
        else float("nan")
        for f in FOLDS
    }

    # Bootstrap p-values per fold (paired against canonical per-fold mean).
    per_fold_p = {
        f: _bootstrap_p_one_sided(
            per_fold_cells[f], canonical_pf[f], n_resamples=n_bootstrap,
            seed=42 + f,
        )
        for f in FOLDS
    }
    # Pool-level p-value (against canonical pool reference).
    pool_p = _bootstrap_p_one_sided(
        all_cells, canonical_pool, n_resamples=n_bootstrap, seed=99,
    )

    return {
        "universe": universe,
        "n_cells": len(all_cells),
        "matrix": matrix,
        "per_fold_mean": per_fold_means,
        "per_fold_sd": per_fold_sds,
        "per_fold_delta_vs_canonical": per_fold_delta,
        "per_fold_p_value": per_fold_p,
        "pool_mean": pool_mean,
        "pool_sd": pool_sd,
        "pool_sos": pool_sos,
        "canonical_pool": canonical_pool,
        "pool_delta_vs_canonical": pool_delta,
        "pool_p_value": pool_p,
    }


def _verdict_block(
    universe: str, summary: Dict[str, object],
) -> Dict[str, object]:
    """Apply the pre-registered success bands for the universe."""
    pool_mean = float(summary["pool_mean"])
    canonical_pool = float(summary["canonical_pool"])
    delta = pool_mean - canonical_pool

    pool_pass = (delta >= -0.05) and (pool_mean == pool_mean)
    pool_hard_fail = delta < -0.10

    if universe == "nasdaq100":
        f2_mean = float(summary["per_fold_mean"][2])
        canonical_f2 = CANONICAL_PER_FOLD[universe][2]
        f2_guard = (f2_mean - canonical_f2) >= -0.20
        guard_name = "F2_>=_canonical_F2_-_0.20"
        guard_value = f2_mean
        guard_canonical = canonical_f2
    elif universe == "biotech_nbi_enriched":
        f4_mean = float(summary["per_fold_mean"][4])
        f5_mean = float(summary["per_fold_mean"][5])
        f4_canon = CANONICAL_PER_FOLD[universe][4]
        f5_canon = CANONICAL_PER_FOLD[universe][5]
        f45_sum = f4_mean + f5_mean
        f45_canon_sum = f4_canon + f5_canon
        f2_guard = (f45_sum - f45_canon_sum) >= -0.20
        guard_name = "F4+F5_>=_canonical_F4+F5_-_0.20"
        guard_value = f45_sum
        guard_canonical = f45_canon_sum
    else:
        raise ValueError(f"unknown universe={universe!r}")

    if pool_hard_fail:
        verdict = "FAIL_HARD"
    elif pool_pass and f2_guard:
        verdict = "PASS"
    elif pool_pass and not f2_guard:
        verdict = "PASS_POOL_FAIL_GUARD"
    elif not pool_pass:
        verdict = "FAIL_POOL"
    else:
        verdict = "UNKNOWN"
    return {
        "verdict": verdict,
        "pool_delta": delta,
        "pool_pass": pool_pass,
        "pool_hard_fail": pool_hard_fail,
        "guard_name": guard_name,
        "guard_value": guard_value,
        "guard_canonical": guard_canonical,
        "guard_pass": f2_guard,
    }


def _format_per_fold_table(
    universe: str, summary: Dict[str, object],
) -> str:
    """Markdown table: canonical vs DSL, with delta + p-value per fold."""
    pf_mean = summary["per_fold_mean"]
    pf_sd = summary["per_fold_sd"]
    pf_p = summary["per_fold_p_value"]
    canonical_pf = CANONICAL_PER_FOLD[universe]
    lines = []
    lines.append(
        f"### Per-fold ({universe}) DSL vs canonical SAC L/S\n"
    )
    lines.append(
        "| fold | canonical | DSL mean | DSL sd | delta | "
        "bootstrap p (one-sided, vs canonical) |"
    )
    lines.append(
        "|------|----------:|---------:|-------:|------:|-------------------------------------:|"
    )
    for f in FOLDS:
        m = pf_mean[f]
        sd = pf_sd[f]
        d = m - canonical_pf[f] if m == m else float("nan")
        p = pf_p[f]
        m_s = f"{m:+.4f}" if m == m else "  NaN "
        sd_s = f"{sd:.4f}" if sd == sd else "  NaN "
        d_s = f"{d:+.4f}" if d == d else "  NaN "
        p_s = f"{p:.4f}" if p == p else "  NaN "
        lines.append(
            f"| F{f} | {canonical_pf[f]:+.4f} | {m_s} | {sd_s} | {d_s} | {p_s} |"
        )
    return "\n".join(lines)


def _format_pool_table(
    summaries: Dict[str, Dict[str, object]],
) -> str:
    """Cross-universe pool summary table."""
    lines = []
    lines.append("## Cross-universe pool summary\n")
    lines.append(
        "| universe              | canonical pool | DSL pool | DSL sd  | "
        "pool delta | bootstrap p | verdict |"
    )
    lines.append(
        "|-----------------------|---------------:|---------:|--------:|"
        "-----------:|------------:|--------:|"
    )
    for u, s in summaries.items():
        canon = float(s["canonical_pool"])
        pool = float(s["pool_mean"])
        sd = float(s["pool_sd"])
        delta = pool - canon if pool == pool else float("nan")
        p = float(s["pool_p_value"])
        v = s["verdict_block"]["verdict"]
        pool_s = f"{pool:+.4f}" if pool == pool else "  NaN "
        sd_s = f"{sd:.4f}" if sd == sd else "  NaN "
        d_s = f"{delta:+.4f}" if delta == delta else "  NaN "
        p_s = f"{p:.4f}" if p == p else "  NaN "
        lines.append(
            f"| {u:<21} | {canon:+.4f} | {pool_s} | {sd_s} | "
            f"{d_s} | {p_s} | {v} |"
        )
    return "\n".join(lines)


def _cross_universe_verdict(
    summaries: Dict[str, Dict[str, object]],
) -> str:
    """Aggregate verdict across universes."""
    pass_count = sum(
        1 for s in summaries.values()
        if s["verdict_block"]["verdict"] == "PASS"
    )
    if pass_count == 2:
        return "STRONG_CROSS_UNIVERSE_GENERALIZATION"
    if pass_count == 1:
        return "PARTIAL_CROSS_UNIVERSE_GENERALIZATION"
    return "SP500_CONDITIONAL_FAIL_BOTH"


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Roll up the DSL NDX + NBI Phase 2 sweep into a 25-cell "
            "summary report. Mirrors rollup_sia_phase3_cross_universe."
        )
    )
    p.add_argument(
        "--out", type=str,
        default="reports/generalizable/dsl_ndx_nbi_2026-05-26.md",
        help="Markdown report destination.",
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=10_000,
        help="Bootstrap resample count (default 10_000).",
    )
    p.add_argument(
        "--nasdaq100-dir", type=str,
        default=str(DEFAULT_DSL_DIR["nasdaq100"]),
    )
    p.add_argument(
        "--biotech-nbi-enriched-dir", type=str,
        default=str(DEFAULT_DSL_DIR["biotech_nbi_enriched"]),
    )
    args = p.parse_args()

    universe_dirs = {
        "nasdaq100": Path(args.nasdaq100_dir),
        "biotech_nbi_enriched": Path(args.biotech_nbi_enriched_dir),
    }

    summaries: Dict[str, Dict[str, object]] = {}
    for u, root in universe_dirs.items():
        s = _per_universe_summary(
            u, root, n_bootstrap=args.n_bootstrap,
        )
        s["verdict_block"] = _verdict_block(u, s)
        summaries[u] = s

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cross = _cross_universe_verdict(summaries)

    lines = []
    lines.append("# DSL Phase 2 cross-universe (NDX + NBI) sweep summary\n")
    lines.append(
        "Differentiable Sharpe surrogate (Option C, commit 6f5eaee) "
        "applied at L1 finetune; identical to SP500 DSL Phase 2 except "
        "for the universe panel + per-universe K (NDX K=20, NBI K=25).\n"
    )
    lines.append(_format_pool_table(summaries))
    lines.append("")
    for u in ("nasdaq100", "biotech_nbi_enriched"):
        lines.append(_format_per_fold_table(u, summaries[u]))
        lines.append("")
    lines.append("## Cross-universe verdict\n")
    lines.append(f"- aggregate verdict: **{cross}**")
    lines.append("")
    lines.append("Per-universe verdict criteria (pre-registered):")
    lines.append(
        "- NDX-100: PASS iff pool >= canonical - 0.05 AND "
        "F2 >= canonical F2 - 0.20"
    )
    lines.append(
        "- NBI-enr: PASS iff pool >= canonical - 0.05 AND "
        "F4+F5 >= canonical F4+F5 - 0.20"
    )
    lines.append(
        "- Either universe: FAIL_HARD if pool delta < -0.10"
    )
    lines.append("")
    lines.append("## Raw per-cell matrix\n")
    for u, s in summaries.items():
        lines.append(f"### {u}\n")
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
    print(f"[rollup_dsl_ndx_nbi] wrote {out_path}")
    print(f"[rollup_dsl_ndx_nbi] cross-universe verdict: {cross}")
    for u, s in summaries.items():
        vb = s["verdict_block"]
        pool = s["pool_mean"]
        canon = s["canonical_pool"]
        d = pool - canon if pool == pool else float("nan")
        print(
            f"  {u}: pool={pool:+.4f} (canonical {canon:+.4f}, "
            f"delta {d:+.4f}) verdict={vb['verdict']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
