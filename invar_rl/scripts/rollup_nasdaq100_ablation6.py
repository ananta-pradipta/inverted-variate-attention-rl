"""Roll up NASDAQ-100 Ablation 6 (macro-conditional concentration) results.

Cross-universe replication of :mod:`invar_rl.scripts.rollup_sp500_ablation6`
on the NDX-100 panel. Reads per-cell strategy-return + K_t parquets at
``outputs/nasdaq100/ablations/ablation6/foldF_seedS.parquet`` and the
per-cell summary JSONs at
``outputs/nasdaq100/ablations/ablation6/summary/foldF_seedS.json``, then
aggregates:

1. Per-fold and pooled annualised Sharpe via two pooling formulas
   (PRIMARY = per-cell mean Sharpe across the 25 cells; secondary =
   day-stream pooled Sharpe with every cell's daily strategy returns
   concatenated, then one Sharpe). Matches the convention used by
   :mod:`invar_rl.scripts.rollup_sp500_ablation6` and the NDX Phase 5
   report at ``reports/nasdaq100/phase_5_layer3_sharpe.md``.
2. Per-fold mean K_t (what concentration did the 2-D SAC pick on
   average in each macro regime?).
3. K_t vs VIX-like signal correlation per fold and pooled. The VIX
   proxy is the bridge's ``avg_corr_z`` (cross-sectional average
   pairwise correlation z-score), aligned to the per-day K_t stream.
   Positive correlation -> SAC concentrates LESS in stressed regimes
   (high VIX -> larger K, more diversified); negative correlation ->
   SAC concentrates MORE in stress (smaller K).

If a cell parquet is missing, it contributes 0 cells to the mean (the
table cell shows ``n=0``); this lets the rollup run incrementally as
the sbatch fleet finishes.

Writes a markdown report to ``reports/nasdaq100/ablation6_concentration.md``.

Usage::

    PYTHONPATH=$PWD python3 -m invar_rl.scripts.rollup_nasdaq100_ablation6
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_TRADING_DAYS = 252
_FILENAME_RE = re.compile(r"^fold(\d+)_seed(\d+)\.parquet$")
_FOLDS = (1, 2, 3, 4, 5)
_SEEDS = (42, 43, 44, 45, 46)

# Canonical NDX-100 Phase 5 L/S (fixed K=20 SAC) reference number from
# drafts/invar_rl_nasdaq100_audit_2026-05-23.md. PRIMARY comparator is
# per-cell mean Sharpe across the 25 (fold, seed) cells, matching
# reports/nasdaq100/phase_5_layer3_sharpe.md.
_CANONICAL_FIXED_K_SHARPE = 1.194
_CANONICAL_KIND = "L/S K=20 (Phase 5 per-cell mean)"


def _annualised_sharpe(rets: np.ndarray) -> float:
    if rets.size < 2:
        return 0.0
    sd = float(rets.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(rets.mean() / sd * np.sqrt(_TRADING_DAYS))


def _load_cells(root: Path) -> List[Dict[str, object]]:
    """Load every ablation6 cell parquet under ``root``."""
    cells: List[Dict[str, object]] = []
    if not root.exists():
        return cells
    summary_dir = root / "summary"
    for p in sorted(root.glob("fold*_seed*.parquet")):
        m = _FILENAME_RE.match(p.name)
        if m is None:
            continue
        fold = int(m.group(1))
        seed = int(m.group(2))
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if df.empty or "strategy_return" not in df.columns:
            continue
        rets = df["strategy_return"].to_numpy(dtype=np.float64)
        ks = (
            df["k_t"].to_numpy(dtype=np.int64)
            if "k_t" in df.columns
            else np.full(rets.shape, -1, dtype=np.int64)
        )
        exps = (
            df["exposure"].to_numpy(dtype=np.float64)
            if "exposure" in df.columns
            else np.zeros_like(rets)
        )
        dates = (
            pd.to_datetime(df["date"]).to_numpy()
            if "date" in df.columns
            else None
        )
        summary_path = summary_dir / f"fold{fold}_seed{seed}.json"
        summary: Optional[Dict] = None
        if summary_path.exists():
            try:
                with open(summary_path) as fh:
                    summary = json.load(fh)
            except Exception:
                summary = None
        cells.append({
            "fold": fold, "seed": seed, "n": int(rets.size),
            "sharpe": _annualised_sharpe(rets),
            "returns": rets,
            "k_t": ks,
            "exposure": exps,
            "dates": dates,
            "path": str(p),
            "summary": summary,
        })
    return cells


def _per_cell_mean_sharpe(
    cells: List[Dict[str, object]],
) -> Tuple[float, float, int]:
    if not cells:
        return 0.0, 0.0, 0
    vals = [float(c["sharpe"]) for c in cells]
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, s, len(vals)


def _day_stream_pooled_sharpe(cells: List[Dict[str, object]]) -> float:
    if not cells:
        return 0.0
    all_rets = np.concatenate([c["returns"] for c in cells])
    return _annualised_sharpe(all_rets)


def _by_fold(cells: List[Dict[str, object]]) -> Dict[int, List[Dict]]:
    out: Dict[int, List[Dict]] = {}
    for c in cells:
        out.setdefault(int(c["fold"]), []).append(c)
    return out


def _avg_corr_z_for_fold(fold: int) -> Optional[Tuple[np.ndarray, object]]:
    """Build the NDX-100 bridge for one fold and return avg_corr_z.

    Lazy import to keep the rollup runnable without bridge dependencies
    when only Sharpe + K_t analysis is needed.
    """
    try:
        import torch
        from src.invar import InVARConfig
        from invar_rl.data.lattice_bridge import build_lattice_bridge
        cfg = InVARConfig(fold=fold, seed=42)
        cfg.panel_kind = "nasdaq100"
        cfg.two_regime_val = True
        cfg.panel_end = "2025-12-31"
        cfg.enable_retrieval_bank = False
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        bridge = build_lattice_bridge(cfg, device=device)
        return bridge.avg_corr_z.astype(np.float64), bridge
    except Exception as exc:
        print(f"[rollup_ablation6] WARN: avg_corr_z unavailable for fold={fold}: {exc}")
        return None


def _k_vix_correlation(
    cells: List[Dict[str, object]],
) -> Dict[str, object]:
    """Pearson + Spearman corr of K_t vs avg_corr_z, per fold and pooled.

    Returns dict with per-fold {pearson, spearman, n_days} and a pooled
    block. Skips folds where either K_t or avg_corr_z is degenerate.
    """
    per_fold: Dict[int, Dict[str, float]] = {}
    all_k: List[float] = []
    all_v: List[float] = []
    fold_groups = _by_fold(cells)
    for fold in sorted(fold_groups):
        out = _avg_corr_z_for_fold(fold)
        if out is None:
            per_fold[fold] = {"pearson": 0.0, "spearman": 0.0, "n_days": 0}
            continue
        avg_corr_z, bridge = out
        dates_arr = np.array(bridge.dates)
        date_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates_arr)}

        ks_concat: List[float] = []
        vs_concat: List[float] = []
        for c in fold_groups[fold]:
            if c["dates"] is None:
                continue
            for date, k in zip(c["dates"], c["k_t"]):
                key = pd.Timestamp(date).normalize()
                if key in date_to_idx:
                    idx = date_to_idx[key]
                    v = float(avg_corr_z[idx])
                    if np.isfinite(v):
                        ks_concat.append(float(k))
                        vs_concat.append(v)
        if len(ks_concat) < 2:
            per_fold[fold] = {"pearson": 0.0, "spearman": 0.0, "n_days": 0}
            continue
        k_arr = np.asarray(ks_concat, dtype=np.float64)
        v_arr = np.asarray(vs_concat, dtype=np.float64)
        if k_arr.std() <= 1e-12 or v_arr.std() <= 1e-12:
            per_fold[fold] = {
                "pearson": 0.0,
                "spearman": 0.0,
                "n_days": int(len(ks_concat)),
            }
        else:
            pearson = float(np.corrcoef(k_arr, v_arr)[0, 1])
            rk = np.argsort(np.argsort(k_arr)).astype(np.float64)
            rv = np.argsort(np.argsort(v_arr)).astype(np.float64)
            spearman = float(np.corrcoef(rk, rv)[0, 1])
            per_fold[fold] = {
                "pearson": pearson,
                "spearman": spearman,
                "n_days": int(len(ks_concat)),
            }
        all_k.extend(ks_concat)
        all_v.extend(vs_concat)

    pooled: Dict[str, float] = {"pearson": 0.0, "spearman": 0.0, "n_days": 0}
    if len(all_k) >= 2:
        k_arr = np.asarray(all_k, dtype=np.float64)
        v_arr = np.asarray(all_v, dtype=np.float64)
        if k_arr.std() > 1e-12 and v_arr.std() > 1e-12:
            pooled["pearson"] = float(np.corrcoef(k_arr, v_arr)[0, 1])
            rk = np.argsort(np.argsort(k_arr)).astype(np.float64)
            rv = np.argsort(np.argsort(v_arr)).astype(np.float64)
            pooled["spearman"] = float(np.corrcoef(rk, rv)[0, 1])
        pooled["n_days"] = int(len(all_k))
    return {"per_fold": per_fold, "pooled": pooled}


def _render_markdown(
    cells: List[Dict[str, object]],
    out_path: Path,
) -> Dict[str, object]:
    """Render the markdown report and return a small summary dict."""
    lines: List[str] = []
    lines.append("# NASDAQ-100 Ablation 6: Macro-Conditional Concentration (2-D SAC)")
    lines.append("")
    lines.append(
        "Cross-universe replication of SP500 Ablation 6 on the NDX-100 panel. "
        "Tests whether a 2-D SAC action (exposure scalar + concentration "
        "K_t via sigmoid logit) beats the canonical NDX-100 1-D exposure "
        "scalar with fixed K=20 per side. Mirrors the Option A canonical "
        "pipeline (InVAR Layer 1 + fixed equal-weight L/S wrapper + Layer 2 "
        "SAC); only the SAC action changes."
    )
    lines.append("")
    lines.append("## Action space")
    lines.append("")
    lines.append("- Canonical (Option A, NDX-100): action = (exposure) in [0, 1.5], 1-D, fixed K=20.")
    lines.append(
        "- Ablation 6: action = (exposure, k_logit) in [0, 1.5] x [-5, 5], "
        "2-D. K_t = round(3 + 17 * sigmoid(k_logit)), clipped to [3, 20]."
    )
    lines.append("")
    lines.append("## Pooling formula")
    lines.append("")
    lines.append(
        "PRIMARY = per-cell annualised Sharpe (sqrt(252) scaling) averaged "
        "across the 25 (fold, seed) cells. Secondary = day-stream pooled "
        "Sharpe with every cell's daily strategy return concatenated. "
        "Matches the convention in reports/nasdaq100/phase_5_layer3_sharpe.md."
    )
    lines.append("")

    fold_groups = _by_fold(cells)
    overall_per_cell_mean, overall_per_cell_std, overall_n = _per_cell_mean_sharpe(cells)
    overall_ds_pooled = _day_stream_pooled_sharpe(cells)

    # Per-fold Sharpe + K_t table.
    lines.append("## Per-fold Sharpe and mean K_t")
    lines.append("")
    lines.append(
        "| fold | n_cells | per-cell mean Sharpe |   std | day-stream pooled | mean K_t | std K_t |"
    )
    lines.append(
        "|-----:|--------:|---------------------:|------:|------------------:|---------:|--------:|"
    )
    fold_summaries: List[Tuple[int, int, float, float, float, float, float]] = []
    for fold in sorted(fold_groups):
        fold_cells = fold_groups[fold]
        m, s, n = _per_cell_mean_sharpe(fold_cells)
        ds = _day_stream_pooled_sharpe(fold_cells)
        ks_concat = np.concatenate([c["k_t"] for c in fold_cells]) if fold_cells else np.array([])
        ks_valid = ks_concat[ks_concat > 0]
        mean_k = float(ks_valid.mean()) if ks_valid.size else 0.0
        std_k = float(ks_valid.std(ddof=1)) if ks_valid.size > 1 else 0.0
        lines.append(
            f"| {fold:>4} | {n:>7} | {m:+.3f}             | {s:.3f} | "
            f"{ds:+.3f}            | {mean_k:>8.2f} | {std_k:>7.2f} |"
        )
        fold_summaries.append((fold, n, m, s, ds, mean_k, std_k))
    overall_k = np.concatenate([c["k_t"] for c in cells]) if cells else np.array([])
    overall_k_valid = overall_k[overall_k > 0]
    overall_mean_k = float(overall_k_valid.mean()) if overall_k_valid.size else 0.0
    overall_std_k = float(overall_k_valid.std(ddof=1)) if overall_k_valid.size > 1 else 0.0
    lines.append(
        f"| pool | {overall_n:>7} | {overall_per_cell_mean:+.3f}             | "
        f"{overall_per_cell_std:.3f} | {overall_ds_pooled:+.3f}            | "
        f"{overall_mean_k:>8.2f} | {overall_std_k:>7.2f} |"
    )
    lines.append("")

    # Vs canonical fixed K=20.
    delta = overall_per_cell_mean - _CANONICAL_FIXED_K_SHARPE
    lines.append("## Vs canonical (Option A) fixed K=20 SAC")
    lines.append("")
    lines.append(
        f"Canonical reference: {_CANONICAL_KIND} = {_CANONICAL_FIXED_K_SHARPE:+.3f} "
        "(drafts/invar_rl_nasdaq100_audit_2026-05-23.md)."
    )
    lines.append("")
    lines.append(
        f"| metric                        | canonical (fixed K=20) | Ablation 6 (macro-conditional K) |  delta |"
    )
    lines.append(
        f"|-------------------------------|-----------------------:|---------------------------------:|-------:|"
    )
    lines.append(
        f"| per-cell mean pooled Sharpe   | {_CANONICAL_FIXED_K_SHARPE:+.3f}                 | "
        f"{overall_per_cell_mean:+.3f}                            | {delta:+.3f} |"
    )
    lines.append("")

    # K_t vs VIX correlation.
    lines.append("## K_t vs VIX-proxy (avg_corr_z) correlation")
    lines.append("")
    lines.append(
        "Positive Pearson/Spearman => SAC picks larger K (more "
        "diversified) when avg_corr_z is high (stress regime); negative "
        "=> SAC concentrates more during stress."
    )
    lines.append("")
    corr = _k_vix_correlation(cells)
    lines.append("| fold | n_days | Pearson | Spearman |")
    lines.append("|-----:|-------:|--------:|---------:|")
    for fold in sorted(corr["per_fold"]):
        c = corr["per_fold"][fold]
        lines.append(
            f"| {fold:>4} | {c['n_days']:>6} | {c['pearson']:+.3f} | {c['spearman']:+.3f} |"
        )
    pooled = corr["pooled"]
    lines.append(
        f"| pool | {pooled['n_days']:>6} | {pooled['pearson']:+.3f} | {pooled['spearman']:+.3f} |"
    )
    lines.append("")

    lines.append("## Per-cell drill-down")
    lines.append("")
    lines.append("| fold | seed | n_days | Sharpe | mean_K | std_K |")
    lines.append("|-----:|-----:|-------:|-------:|-------:|------:|")
    for c in sorted(cells, key=lambda x: (x["fold"], x["seed"])):
        k_valid = c["k_t"][c["k_t"] > 0]
        mk = float(k_valid.mean()) if k_valid.size else 0.0
        sk = float(k_valid.std(ddof=1)) if k_valid.size > 1 else 0.0
        lines.append(
            f"| {c['fold']:>4} | {c['seed']:>4} | {c['n']:>6} | "
            f"{c['sharpe']:+.3f} | {mk:>6.2f} | {sk:>5.2f} |"
        )
    lines.append("")

    lines.append("## Interpretation rubric")
    lines.append("")
    lines.append(
        "- If pooled Sharpe is at least +0.05 above the canonical "
        f"{_CANONICAL_FIXED_K_SHARPE:+.3f}: macro-conditional K adds value; "
        "reinforces the Option A macro-aware-controller framing and "
        "motivates upgrading the action space."
    )
    lines.append(
        "- If pooled Sharpe is within +/-0.05 of the canonical: parsimony "
        "wins; the 1-D exposure scalar is sufficient, fixed K=20 stays "
        "canonical, K_t adds noise but no signal."
    )
    lines.append(
        "- If pooled Sharpe is below the canonical by more than 0.05: "
        "the extra action degree of freedom hurts; SAC fails to learn "
        "useful concentration timing at this training budget."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))

    return {
        "n_cells": int(overall_n),
        "per_cell_mean_sharpe": float(overall_per_cell_mean),
        "per_cell_std_sharpe": float(overall_per_cell_std),
        "day_stream_pooled_sharpe": float(overall_ds_pooled),
        "delta_vs_canonical_fixed_k": float(delta),
        "mean_k": float(overall_mean_k),
        "std_k": float(overall_std_k),
        "k_vix_pooled": corr["pooled"],
    }


def main() -> int:
    root = Path("outputs/nasdaq100/ablations/ablation6")
    cells = _load_cells(root)
    out_path = Path("reports/nasdaq100/ablation6_concentration.md")
    summary = _render_markdown(cells, out_path)
    print(
        f"[rollup ablation6] wrote {out_path}: n_cells={summary['n_cells']} "
        f"per-cell={summary['per_cell_mean_sharpe']:+.4f} "
        f"+/- {summary['per_cell_std_sharpe']:.4f} "
        f"day-stream={summary['day_stream_pooled_sharpe']:+.4f} "
        f"delta_vs_canonical_fixed_k=({summary['delta_vs_canonical_fixed_k']:+.4f}) "
        f"mean_K={summary['mean_k']:.2f}"
    )
    print(
        f"  K_t vs VIX (pooled): pearson={summary['k_vix_pooled']['pearson']:+.3f} "
        f"spearman={summary['k_vix_pooled']['spearman']:+.3f} "
        f"n_days={summary['k_vix_pooled']['n_days']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
