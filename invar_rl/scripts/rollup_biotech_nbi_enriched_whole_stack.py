"""Roll up biotech NBI enriched-panel whole-stack RL baselines.

Reads per-cell JSONs at
``outputs/biotech_nbi_enriched/whole_stack_rl/{finrl/finrl_{ppo,a2c,ddpg},stockformer}/foldF_seedS.json``
and reports per-cell mean annualised Sharpe across the 25 (fold, seed)
cells per method. PRIMARY = per-cell mean Sharpe.

The cell JSONs are produced by
:mod:`invar_rl.training.finrl_faithful_eval` (``--phase
biotech_nbi_enriched``) and
:mod:`invar_rl.training.stockformer_faithful_eval` (``--phase
biotech_nbi_enriched``); both write a top-level ``perf`` dict containing
``sharpe_annualised``.

Usage::

    PYTHONPATH=$PWD python3 -m invar_rl.scripts.rollup_biotech_nbi_enriched_whole_stack
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple


_DEFAULT_FINRL_ROOT = Path(
    "outputs/biotech_nbi_enriched/whole_stack_rl/finrl"
)
_DEFAULT_SF_ROOT = Path(
    "outputs/biotech_nbi_enriched/whole_stack_rl/stockformer"
)
_DEFAULT_OUT = Path(
    "reports/biotech_nbi/enriched_whole_stack_rl.md"
)

_FINRL_ALGOS: Tuple[str, ...] = ("ppo", "a2c", "ddpg")


def _load_cells(folder: Path) -> List[dict]:
    cells: List[dict] = []
    if not folder.exists():
        return cells
    for p in sorted(folder.glob("fold*_seed*.json")):
        try:
            with open(p) as fh:
                cells.append(json.load(fh))
        except Exception:
            continue
    return cells


def _sharpe_from(cell: dict) -> Optional[float]:
    perf = cell.get("perf") or {}
    v = perf.get("sharpe_annualised")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pooled(cells: List[dict]) -> Tuple[int, float, float]:
    vals = [v for v in (_sharpe_from(c) for c in cells) if v is not None]
    if not vals:
        return 0, 0.0, 0.0
    m = float(mean(vals))
    s = float(stdev(vals)) if len(vals) > 1 else 0.0
    return len(vals), m, s


def _per_fold(cells: List[dict]) -> List[Tuple[int, int, float, float]]:
    by_fold: Dict[int, List[float]] = {}
    for c in cells:
        v = _sharpe_from(c)
        if v is None:
            continue
        f = int(c.get("fold", 0))
        by_fold.setdefault(f, []).append(v)
    rows: List[Tuple[int, int, float, float]] = []
    for f in sorted(by_fold):
        vals = by_fold[f]
        m = float(mean(vals))
        s = float(stdev(vals)) if len(vals) > 1 else 0.0
        rows.append((f, len(vals), m, s))
    return rows


def _render(
    finrl_root: Path, sf_root: Path, out_path: Path,
) -> Dict[str, Tuple[int, float, float]]:
    method_grids: Dict[str, List[dict]] = {}
    for algo in _FINRL_ALGOS:
        method_grids[f"finrl_{algo}"] = _load_cells(finrl_root / f"finrl_{algo}")
    method_grids["stockformer"] = _load_cells(sf_root)

    pretty: Dict[str, str] = {
        "finrl_ppo": "FinRL PPO (whole-stack RL)",
        "finrl_a2c": "FinRL A2C (whole-stack RL)",
        "finrl_ddpg": "FinRL DDPG (whole-stack RL)",
        "stockformer": "StockFormer (faithful, whole-stack RL)",
    }

    lines: List[str] = []
    lines.append("# Biotech NBI enriched: whole-stack RL baselines")
    lines.append("")
    lines.append(
        "Closes fairness audit issue #5. FinRL (PPO / A2C / DDPG) and "
        "StockFormer were originally trained on the 26-feature zero-fill "
        "biotech NBI panel (Phase 5.5 baselines, "
        "`outputs/biotech_nbi/baselines/`). Headline ranker comparisons "
        "moved to the 22-feature enriched panel "
        "(`outputs/biotech_nbi_enriched/baselines/`) but whole-stack RL "
        "stayed on zero-fill: an apples-to-oranges fairness gap. This "
        "rollup re-reports FinRL + StockFormer on the enriched panel "
        "(same hyperparameters as the zero-fill runs, only panel changed)."
    )
    lines.append("")
    lines.append("## Per-cell mean annualised Sharpe (PRIMARY)")
    lines.append("")
    lines.append("| method | n_cells | mean Sharpe | std |")
    lines.append("|---|---:|---:|---:|")
    summary: Dict[str, Tuple[int, float, float]] = {}
    for key in ("finrl_ppo", "finrl_a2c", "finrl_ddpg", "stockformer"):
        n, m, s = _pooled(method_grids[key])
        summary[key] = (n, m, s)
        label = pretty[key]
        if n == 0:
            lines.append(f"| {label} | 0 | -- | -- |")
        else:
            lines.append(f"| {label} | {n} | {m:+.4f} | {s:.4f} |")
    lines.append("")
    lines.append("## Per-fold drill-down (appendix)")
    lines.append("")
    for key in ("finrl_ppo", "finrl_a2c", "finrl_ddpg", "stockformer"):
        rows = _per_fold(method_grids[key])
        if not rows:
            continue
        lines.append(f"### {pretty[key]}")
        lines.append("")
        lines.append("| fold | n_seeds | mean | std |")
        lines.append("|---:|---:|---:|---:|")
        for f, n, m, s in rows:
            lines.append(f"| {f} | {n} | {m:+.4f} | {s:.4f} |")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Folds 1..5, seeds 42..46, 25 cells per method (worst case)."
    )
    lines.append(
        "- Pooling formula matches `rollup_biotech_nbi_baselines.py`: "
        "per-cell annualised Sharpe averaged across the 25 cells. "
        "Per-cell Sharpe is computed inside the eval driver from the "
        "test-window daily return stream."
    )
    lines.append(
        "- FinRL hyperparameters: total_timesteps=50,000 per cell, "
        "FinRLEnvConfig defaults (turbulence_threshold=140, hmax=100, "
        "initial_balance=1e6, tx cost 0.1%, reward_scaling=1e-4)."
    )
    lines.append(
        "- StockFormer hyperparameters: total_timesteps=10,000 per cell, "
        "pretrain_epochs=30, pretrain_batch_size=32, pretrain_lr=1e-4, "
        "sac_lr=1e-4, universe_k=30."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Roll up biotech NBI enriched-panel whole-stack RL "
            "(FinRL + StockFormer) baselines."
        )
    )
    parser.add_argument(
        "--finrl-root", type=str, default=str(_DEFAULT_FINRL_ROOT),
    )
    parser.add_argument(
        "--stockformer-root", type=str, default=str(_DEFAULT_SF_ROOT),
    )
    parser.add_argument(
        "--out", type=str, default=str(_DEFAULT_OUT),
    )
    args = parser.parse_args()
    finrl_root = Path(args.finrl_root)
    sf_root = Path(args.stockformer_root)
    out_path = Path(args.out)
    summary = _render(finrl_root, sf_root, out_path)
    print(f"[rollup whole-stack RL enriched] wrote {out_path}")
    for key, (n, m, s) in summary.items():
        print(f"  {key:18s} n={n:>2} mean={m:+.4f} std={s:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
