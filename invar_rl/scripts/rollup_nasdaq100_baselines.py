"""Roll up NASDAQ-100 Phase 5.5 baseline results.

Reads:
  - outputs/nasdaq100/baselines/{master,factorvae,stockmixer,dystage,swa_invar}/foldF_seedS.json
    (Layer-1 ranker baselines; both sharpe_ls and sharpe_lo_native per cell)
  - outputs/nasdaq100/baselines/{finrl_ppo,finrl_a2c,finrl_ddpg}/foldF_seedS.json
    (FinRL whole-stack RL; long-only top-30 active)
  - outputs/nasdaq100/baselines/stockformer/foldF_seedS.json
    (StockFormer faithful; long-only top-30 active)
  - outputs/nasdaq100/baselines/non_learning/{strategy}.json
    (5 deterministic non-learning baselines)

Aggregates into:
  - per-fold mean (+/- std over seeds) and pooled across all 25 cells per
    Layer-1 ranker baseline, per protocol
  - per-fold mean (+/- std over seeds) and pooled across all 25 cells per
    whole-stack RL baseline (long-only)
  - per-fold Sharpe + pooled day-stream Sharpe per non-learning strategy

Writes:
  - reports/nasdaq100/phase_5_5_baselines.md
    (Panel A: ranker baselines under L/S top-25 wrapper and native long-only
                top-K; Panel B: whole-stack RL long-only; Panel C: non-
                learning baselines.)

Usage::

    PYTHONPATH=$PWD python3 invar_rl/scripts/rollup_nasdaq100_baselines.py
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

import numpy as np


_OUT_ROOT = Path("outputs/nasdaq100/baselines")
_REPORT_PATH = Path("reports/nasdaq100/phase_5_5_baselines.md")

_RANKER_BASELINES: Tuple[str, ...] = (
    "master", "factorvae", "stockmixer", "dystage", "swa_invar",
    "invar_l1",
)
_RANKER_PRETTY: Dict[str, str] = {
    "master": "MASTER (AAAI'24)",
    "factorvae": "FactorVAE (AAAI'22)",
    "stockmixer": "StockMixer (AAAI'24)",
    "dystage": "DySTAGE (ICAIF'24)",
    "swa_invar": "SWA-InVAR (ours)",
    "invar_l1": "InVAR Layer 1 + top-25 L/S wrapper (ours, no QP / no SAC)",
}
_RANKER_NATIVE_K: Dict[str, int] = {
    "master": 30, "factorvae": 50,
    "stockmixer": 25, "dystage": 25, "swa_invar": 25,
    "invar_l1": 25,
}
# Baselines for which a native long-only top-K column is meaningful.
# InVAR Layer 1 + wrapper is L/S only; its sharpe_lo_native field is a
# schema-parity duplicate of sharpe_ls, so we suppress it in the table.
_RANKER_HAS_LO_NATIVE: Dict[str, bool] = {
    "master": True, "factorvae": True, "stockmixer": True,
    "dystage": True, "swa_invar": True, "invar_l1": False,
}

_RL_BASELINES: Tuple[str, ...] = (
    "finrl_ppo", "finrl_a2c", "finrl_ddpg", "stockformer",
)
_RL_PRETTY: Dict[str, str] = {
    "finrl_ppo": "FinRL PPO",
    "finrl_a2c": "FinRL A2C",
    "finrl_ddpg": "FinRL DDPG",
    "stockformer": "StockFormer (faithful)",
}

_NON_LEARNING: Tuple[str, ...] = (
    "buy_and_hold", "equal_weight_long",
    "momentum_jt_12_2", "reversal_1m", "vol_targeted_market_10",
)
_NL_PRETTY: Dict[str, str] = {
    "buy_and_hold": "Buy-and-hold EW",
    "equal_weight_long": "Equal-weight long (daily rebalance)",
    "momentum_jt_12_2": "Jegadeesh-Titman 12-2 momentum (L/S decile)",
    "reversal_1m": "1-month reversal (L/S decile)",
    "vol_targeted_market_10": "Vol-targeted market (10% ann)",
}


def _load_ranker_cells(baseline: str) -> List[dict]:
    """Load all foldF_seedS.json for a Layer-1 ranker baseline."""
    b_dir = _OUT_ROOT / baseline
    if not b_dir.exists():
        return []
    cells: List[dict] = []
    for p in sorted(b_dir.glob("fold*_seed*.json")):
        with open(p) as fh:
            cells.append(json.load(fh))
    return cells


def _load_rl_cells(baseline: str) -> List[dict]:
    """Load all foldF_seedS.json for a whole-stack RL baseline."""
    b_dir = _OUT_ROOT / baseline
    if not b_dir.exists():
        return []
    cells: List[dict] = []
    for p in sorted(b_dir.glob("fold*_seed*.json")):
        with open(p) as fh:
            cells.append(json.load(fh))
    return cells


def _per_fold_stats(
    cells: List[dict], key_extractor
) -> Dict[int, Tuple[int, float, float]]:
    """Group cells by fold and compute (n, mean, std) of the chosen scalar."""
    by_fold: Dict[int, List[float]] = {}
    for c in cells:
        f = int(c["fold"])
        v = float(key_extractor(c))
        by_fold.setdefault(f, []).append(v)
    return {
        f: (len(vals), float(mean(vals)),
            float(stdev(vals)) if len(vals) > 1 else 0.0)
        for f, vals in sorted(by_fold.items())
    }


def _pooled_stats(cells: List[dict], key_extractor) -> Tuple[int, float, float]:
    """Pool the scalar across all cells -> (n, mean, std)."""
    if not cells:
        return 0, 0.0, 0.0
    vals = [float(key_extractor(c)) for c in cells]
    return (
        len(vals), float(mean(vals)),
        float(stdev(vals)) if len(vals) > 1 else 0.0,
    )


def _ranker_section() -> str:
    """Panel A: Layer-1 ranker baselines under both protocols."""
    lines: List[str] = []
    lines.append("## Panel A: Layer-1 ranker baselines (NDX-100)")
    lines.append("")
    lines.append(
        "Five published ranker baselines plus an InVAR Layer 1 ablation "
        "row, trained on the NASDAQ-100 panel "
        "(`panel_kind=nasdaq100`, `two_regime_val=True`, "
        "`panel_end=2025-12-31`), 5 folds x 5 seeds = 25 cells each. "
        "Each cell evaluated under (a) the top-25 L/S wrapper (symmetric "
        "to the S&P 500 Panel A reference) and, for the published "
        "baselines, (b) the authors' native long-only top-K."
    )
    lines.append("")
    lines.append(
        "**Pooling: per-cell mean (PRIMARY).** Annualised Sharpe is "
        "computed per (fold, seed) cell, then averaged across the 25 "
        "cells. Same formula used by the Phase 5 InVAR-RL Layer 3 SAC "
        "rollup in `phase_5_layer3_sharpe.md` (which also reports "
        "day-stream pooled Sharpe as a secondary column)."
    )
    lines.append("")
    lines.append(
        "**InVAR Layer 1 + top-25 L/S wrapper** row uses the trained "
        "canonical InVAR Layer-1 scores under the same wrapper as the "
        "published baselines; NO Layer-2 QP, NO Layer-3 SAC. The "
        "difference vs the Phase 5 Layer-3 SAC L/S row isolates the "
        "marginal contribution of the QP + SAC stack on top of InVAR's "
        "Layer 1 ranker."
    )
    lines.append("")
    lines.append(
        "| baseline | n_cells | L/S top-25 mean (std) | L-only native K mean (std) | native K |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )
    for b in _RANKER_BASELINES:
        cells = _load_ranker_cells(b)
        n_ls, m_ls, s_ls = _pooled_stats(cells, lambda c: c.get("sharpe_ls", 0.0))
        if _RANKER_HAS_LO_NATIVE.get(b, True):
            n_lo, m_lo, s_lo = _pooled_stats(
                cells, lambda c: c.get("sharpe_lo_native", 0.0)
            )
            lo_cell = f"{m_lo:+.3f} ({s_lo:.3f})"
            k_cell = f"{_RANKER_NATIVE_K[b]}"
        else:
            lo_cell = "n/a"
            k_cell = "n/a"
        lines.append(
            f"| {_RANKER_PRETTY[b]} | {n_ls} | "
            f"{m_ls:+.3f} ({s_ls:.3f}) | "
            f"{lo_cell} | {k_cell} |"
        )
    lines.append("")
    # Per-fold breakdowns
    for b in _RANKER_BASELINES:
        cells = _load_ranker_cells(b)
        if not cells:
            continue
        lines.append(f"### {_RANKER_PRETTY[b]} - per-fold breakdown")
        lines.append("")
        ls_pf = _per_fold_stats(cells, lambda c: c.get("sharpe_ls", 0.0))
        if _RANKER_HAS_LO_NATIVE.get(b, True):
            lo_pf = _per_fold_stats(
                cells, lambda c: c.get("sharpe_lo_native", 0.0)
            )
            lines.append(
                "| fold | n | L/S Sharpe (std) | L-only Sharpe (std) |"
            )
            lines.append("|---:|---:|---:|---:|")
            for f in sorted(set(ls_pf) | set(lo_pf)):
                n_ls, m_ls, s_ls = ls_pf.get(f, (0, 0.0, 0.0))
                n_lo, m_lo, s_lo = lo_pf.get(f, (0, 0.0, 0.0))
                lines.append(
                    f"| {f} | {n_ls} | {m_ls:+.3f} ({s_ls:.3f}) | "
                    f"{m_lo:+.3f} ({s_lo:.3f}) |"
                )
        else:
            lines.append("| fold | n | L/S Sharpe (std) |")
            lines.append("|---:|---:|---:|")
            for f in sorted(ls_pf):
                n_ls, m_ls, s_ls = ls_pf.get(f, (0, 0.0, 0.0))
                lines.append(
                    f"| {f} | {n_ls} | {m_ls:+.3f} ({s_ls:.3f}) |"
                )
        lines.append("")
    return "\n".join(lines)


def _rl_section() -> str:
    """Panel B: whole-stack RL baselines (long-only top-30 active)."""
    lines: List[str] = []
    lines.append("## Panel B: Whole-stack RL baselines (NDX-100, long-only top-30 active)")
    lines.append("")
    lines.append(
        "FinRL (PPO, A2C, DDPG) and StockFormer (faithful) trained end-to-end "
        "on the top-30 most-active NDX-100 names per fold, under the "
        "InVAR-RL 5-fold macro-stratified protocol. FinRL: 50k SAC-equivalent "
        "timesteps per cell. StockFormer: 10k SAC + 30 transformer pretrain "
        "epochs per cell (matches the S&P 500 stress-test budget)."
    )
    lines.append("")
    lines.append("| baseline | n_cells | Sharpe mean (std) |")
    lines.append("|---|---:|---:|")
    for b in _RL_BASELINES:
        cells = _load_rl_cells(b)
        n, m, s = _pooled_stats(
            cells, lambda c: c["perf"]["sharpe_annualised"]
            if "perf" in c else c.get("sharpe_annualised", 0.0)
        )
        lines.append(
            f"| {_RL_PRETTY[b]} | {n} | {m:+.3f} ({s:.3f}) |"
        )
    lines.append("")
    for b in _RL_BASELINES:
        cells = _load_rl_cells(b)
        if not cells:
            continue
        lines.append(f"### {_RL_PRETTY[b]} - per-fold breakdown")
        lines.append("")
        pf = _per_fold_stats(
            cells,
            lambda c: c["perf"]["sharpe_annualised"]
            if "perf" in c else c.get("sharpe_annualised", 0.0),
        )
        lines.append("| fold | n_seeds | Sharpe mean (std) |")
        lines.append("|---:|---:|---:|")
        for f, (n, m, s) in pf.items():
            lines.append(f"| {f} | {n} | {m:+.3f} ({s:.3f}) |")
        lines.append("")
    return "\n".join(lines)


def _non_learning_section() -> str:
    """Panel C: 5 non-learning baselines (deterministic)."""
    lines: List[str] = []
    lines.append("## Panel C: Non-learning baselines (NDX-100, CPU-only)")
    lines.append("")
    lines.append(
        "Computed analytically from `data/nasdaq100/prices.parquet` and "
        "`data/nasdaq100/active_mask.parquet`. Deterministic (no seed "
        "loop); per-fold Sharpe over the test segment plus day-stream "
        "pooled Sharpe across all 5 folds' test segments concatenated."
    )
    lines.append("")
    lines.append(
        "| strategy | pooled Sharpe | per-fold mean Sharpe |"
    )
    lines.append("|---|---:|---:|")
    nl_dir = _OUT_ROOT / "non_learning"
    for strat in _NON_LEARNING:
        p = nl_dir / f"{strat}.json"
        if not p.exists():
            lines.append(f"| {_NL_PRETTY[strat]} | n/a | n/a |")
            continue
        with open(p) as fh:
            payload = json.load(fh)
        pooled = float(payload["pooled"]["pooled_sharpe"])
        per_fold = list(payload["per_fold_sharpe"].values())
        per_fold_mean_val = float(np.mean(per_fold)) if per_fold else 0.0
        lines.append(
            f"| {_NL_PRETTY[strat]} | {pooled:+.3f} | "
            f"{per_fold_mean_val:+.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _header() -> str:
    return (
        "# NASDAQ-100 Phase 5.5: baseline replication\n\n"
        "Side-by-side rollup of 5 Layer-1 ranker baselines + an InVAR "
        "Layer 1 ablation + 4 whole-stack RL baselines + 5 non-learning "
        "baselines on the NASDAQ-100 panel, under the InVAR-RL 5-fold "
        "macro-stratified protocol (`panel_kind=nasdaq100`, "
        "`two_regime_val=True`, `panel_end=2025-12-31`). Mirrors the "
        "S&P 500 Phase A/B audit; all baseline architectures and "
        "hyperparameters held byte-for-byte from the S&P 500 protocol "
        "(Policy P1, no retuning).\n\n"
        "All Sharpe numbers in this report use the **per-cell mean** "
        "pooling formula (annualised Sharpe per (fold, seed) cell, then "
        "averaged across cells). The Phase 5 InVAR-RL Layer 3 SAC report "
        "(`phase_5_layer3_sharpe.md`) reports the same formula as its "
        "primary, with day-stream pooled Sharpe as a secondary column.\n\n"
        "Per audit `drafts/invar_rl_nasdaq100_audit_2026-05-23.md`, this "
        "standardisation fixes an earlier apples-to-oranges comparison "
        "between Phase 5 (previously day-stream pooled) and Phase 5.5 "
        "(per-cell mean).\n\n"
    )


def main() -> int:
    sections = [
        _header(),
        _ranker_section(),
        _rl_section(),
        _non_learning_section(),
    ]
    text = "\n".join(sections).rstrip() + "\n"
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(text)
    print(f"[rollup] wrote {_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
