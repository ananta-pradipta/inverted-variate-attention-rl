"""Roll up NASDAQ-100 Phase 5 Layer 3 SAC controller results.

Reads the per-cell strategy-return parquets at
``outputs/nasdaq100/layer3/{ls,lo}/fold*_seed*.parquet`` and aggregates
Sharpe per (protocol, fold) and per-protocol across all 25 cells.

Two pooling formulas are emitted side-by-side:

  * **Per-cell mean** (PRIMARY, matches Phase 5.5 baseline rollup): compute
    annualised Sharpe per (fold, seed) cell, then average across cells.
  * **Day-stream pooled** (SECONDARY): concatenate every cell's daily
    strategy returns into one stream, then compute one annualised Sharpe
    over the concatenated stream.

The two formulas differ structurally when fold-level Sharpe distributions
are skewed (e.g. NDX-100 fold 2 has a heavily negative regime that drags
the day-stream pooled std up). Per-cell mean is the canonical comparator
for cross-baseline tables because Phase 5.5 baselines are reported under
the same formula (see ``invar_rl/training/nasdaq100_baseline_eval.py``).

Writes a markdown summary to
``reports/nasdaq100/phase_5_layer3_sharpe.md``.

Usage::

    PYTHONPATH=$PWD python3 -m invar_rl.scripts.rollup_nasdaq100_layer3
"""
from __future__ import annotations

import re
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_TRADING_DAYS = 252
_FILENAME_RE = re.compile(r"^fold(\d+)_seed(\d+)\.parquet$")


def _annualised_sharpe(rets: np.ndarray) -> float:
    """Annualised Sharpe with sqrt(252) scaling."""
    if rets.size < 2:
        return 0.0
    sd = float(rets.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(rets.mean() / sd * np.sqrt(_TRADING_DAYS))


def _load_protocol(root: Path) -> List[Dict[str, object]]:
    """Load every per-cell parquet under ``root`` and return cell records."""
    cells: List[Dict[str, object]] = []
    for p in sorted(root.glob("fold*_seed*.parquet")):
        m = _FILENAME_RE.match(p.name)
        if m is None:
            continue
        fold = int(m.group(1))
        seed = int(m.group(2))
        df = pd.read_parquet(p)
        if "strategy_return" not in df.columns:
            continue
        rets = df["strategy_return"].to_numpy(dtype=np.float64)
        cells.append({
            "fold": fold, "seed": seed, "n": int(rets.size),
            "sharpe": _annualised_sharpe(rets),
            "returns": rets,
            "path": str(p),
        })
    return cells


def _per_fold_table(
    cells: List[Dict[str, object]]
) -> List[Tuple[int, int, float, float]]:
    """Per-fold (mean +/- std) Sharpe over the seed cells."""
    by_fold: Dict[int, List[float]] = {}
    for c in cells:
        by_fold.setdefault(int(c["fold"]), []).append(float(c["sharpe"]))
    rows: List[Tuple[int, int, float, float]] = []
    for f in sorted(by_fold):
        vals = by_fold[f]
        m = mean(vals)
        s = stdev(vals) if len(vals) > 1 else 0.0
        rows.append((f, len(vals), m, s))
    return rows


def _day_stream_pooled_sharpe(cells: List[Dict[str, object]]) -> float:
    """Pooled Sharpe across all per-cell daily returns concatenated."""
    if not cells:
        return 0.0
    all_rets = np.concatenate([c["returns"] for c in cells])
    return _annualised_sharpe(all_rets)


def _per_cell_mean_sharpe(
    cells: List[Dict[str, object]]
) -> Tuple[float, float]:
    """Mean and std (ddof=1) of per-cell Sharpe."""
    if not cells:
        return 0.0, 0.0
    vals = [float(c["sharpe"]) for c in cells]
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, s


def _render_markdown(
    ls_cells: List[Dict[str, object]],
    lo_cells: List[Dict[str, object]],
    out_path: Path,
) -> None:
    """Write the phase 5 markdown report."""
    lines: List[str] = []
    lines.append("# NASDAQ-100 Phase 5: Layer 3 SAC Sharpe rollup")
    lines.append("")
    lines.append(
        "Per-cell strategy returns are read from "
        "`outputs/nasdaq100/layer3/{ls,lo}/foldF_seedS.parquet`. Two pooling "
        "formulas are reported side-by-side:"
    )
    lines.append("")
    lines.append(
        "- **Per-cell mean (PRIMARY)**: annualised Sharpe is computed per "
        "(fold, seed) cell, then averaged across cells. Matches the Phase "
        "5.5 baseline rollup in `reports/nasdaq100/phase_5_5_baselines.md`."
    )
    lines.append(
        "- **Day-stream pooled (SECONDARY)**: every cell's daily strategy "
        "return is concatenated into one stream; annualised Sharpe is "
        "computed once on the concatenated stream. More sensitive to "
        "fold-level distributional skew (e.g. NDX-100 fold 2)."
    )
    lines.append("")
    lines.append(
        f"Annualisation factor: sqrt({_TRADING_DAYS})."
    )
    lines.append("")

    for label, cells in (("Long-short (L/S)", ls_cells),
                         ("Long-only (L/O)", lo_cells)):
        lines.append(f"## {label}")
        lines.append("")
        if not cells:
            lines.append("(no parquets found)")
            lines.append("")
            continue
        lines.append(f"Cells loaded: {len(cells)}")
        per_fold = _per_fold_table(cells)
        lines.append("")
        lines.append("Per-fold (mean +/- std over seeds):")
        lines.append("")
        lines.append("| fold | n_seeds |   mean | std    |")
        lines.append("|-----:|--------:|-------:|-------:|")
        for f, n, m, s in per_fold:
            lines.append(f"| {f:>4} | {n:>7} | {m:+.4f} | {s:.4f} |")
        lines.append("")
        # Per-seed pooled (mean of per-fold Sharpe within seed).
        by_seed: Dict[int, List[float]] = {}
        for c in cells:
            by_seed.setdefault(int(c["seed"]), []).append(float(c["sharpe"]))
        lines.append("Per-seed pooled (mean over folds within seed):")
        lines.append("")
        lines.append("| seed | n_folds |   mean |")
        lines.append("|-----:|--------:|-------:|")
        for seed in sorted(by_seed):
            vals = by_seed[seed]
            lines.append(
                f"| {seed:>4} | {len(vals):>7} | {mean(vals):+.4f} |"
            )
        lines.append("")
        pc_mean, pc_std = _per_cell_mean_sharpe(cells)
        ds_pooled = _day_stream_pooled_sharpe(cells)
        lines.append("Pooled Sharpe (both formulas):")
        lines.append("")
        lines.append("| pooling formula | Sharpe | std (across cells) |")
        lines.append("|---|---:|---:|")
        lines.append(
            f"| per-cell mean (PRIMARY) | {pc_mean:+.4f} | {pc_std:.4f} |"
        )
        lines.append(
            f"| day-stream pooled (secondary) | {ds_pooled:+.4f} | -- |"
        )
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> int:
    root = Path("outputs/nasdaq100/layer3")
    ls_cells = _load_protocol(root / "ls")
    lo_cells = _load_protocol(root / "lo")
    out_path = Path("reports/nasdaq100/phase_5_layer3_sharpe.md")
    _render_markdown(ls_cells, lo_cells, out_path)
    print(
        f"[rollup] wrote {out_path}: "
        f"ls cells={len(ls_cells)} lo cells={len(lo_cells)}"
    )
    if ls_cells:
        ls_pc_mean, ls_pc_std = _per_cell_mean_sharpe(ls_cells)
        print(
            f"[rollup] LS per-cell mean Sharpe (PRIMARY): "
            f"{ls_pc_mean:+.4f} +/- {ls_pc_std:.4f}"
        )
        print(
            f"[rollup] LS day-stream pooled Sharpe (secondary): "
            f"{_day_stream_pooled_sharpe(ls_cells):+.4f}"
        )
    if lo_cells:
        lo_pc_mean, lo_pc_std = _per_cell_mean_sharpe(lo_cells)
        print(
            f"[rollup] LO per-cell mean Sharpe (PRIMARY): "
            f"{lo_pc_mean:+.4f} +/- {lo_pc_std:.4f}"
        )
        print(
            f"[rollup] LO day-stream pooled Sharpe (secondary): "
            f"{_day_stream_pooled_sharpe(lo_cells):+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
