"""Roll up the NASDAQ-100 Phase 6 four-ablation results into a 4x4 table.

Reads the per-cell strategy-return parquets at
``outputs/nasdaq100/phase6_ablation/{ablation}/{method}/{protocol}/foldF_seedS.parquet``
for the 30 (ablation, method, protocol) tuples produced by
:mod:`invar_rl.training.nasdaq100_ablation_eval` (32 - 2 for the undefined
stripped_l3 + constant_full pair), and aggregates pooled annualised Sharpe
per cell, per fold, and per (ablation, method, protocol).

The headline output is a 4 x 4 ablation table per protocol mirroring the
SP500 Phase 6 Table 4 (rows = methods recurrent_ppo / feedforward_ppo /
sac / constant_full; columns = ablations canonical / random_l1 /
equal_l2 / stripped_l3) with per-cell mean Sharpe in each cell.

Pooling formula (PRIMARY): per-cell annualised Sharpe averaged across
the 25 (fold, seed) cells of each tuple. Matches the Phase 5 +
Phase 5.5 rollup (see ``invar_rl/scripts/rollup_nasdaq100_layer3.py``).
Day-stream pooled Sharpe (concat every cell's daily strategy returns
into one stream, then compute one Sharpe) is reported as a secondary
column.

If a cell parquet is missing, it contributes 0 cells to the mean (the
table cell shows ``n=0``); this lets the rollup run incrementally as
the sbatch fleet finishes.

Writes a markdown report to
``reports/nasdaq100/phase_6_ablations.md``.

Usage::

    PYTHONPATH=$PWD python3 -m invar_rl.scripts.rollup_nasdaq100_ablations
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
_ABLATIONS = ("canonical", "random_l1", "equal_l2", "stripped_l3")
_METHODS = ("recurrent_ppo", "feedforward_ppo", "sac", "constant_full")
_PROTOCOLS = ("ls", "lo")


def _annualised_sharpe(rets: np.ndarray) -> float:
    if rets.size < 2:
        return 0.0
    sd = float(rets.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(rets.mean() / sd * np.sqrt(_TRADING_DAYS))


def _load_tuple_cells(
    root: Path, ablation: str, method: str, protocol: str,
) -> List[Dict[str, object]]:
    """Load every parquet under outputs/.../{ablation}/{method}/{protocol}/."""
    folder = root / ablation / method / protocol
    cells: List[Dict[str, object]] = []
    if not folder.exists():
        return cells
    for p in sorted(folder.glob("fold*_seed*.parquet")):
        m = _FILENAME_RE.match(p.name)
        if m is None:
            continue
        fold = int(m.group(1))
        seed = int(m.group(2))
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
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


def _per_fold_means(
    cells: List[Dict[str, object]],
) -> List[Tuple[int, int, float, float]]:
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


def _format_cell(m: float, s: float, n: int) -> str:
    if n == 0:
        return "-- (n=0)"
    return f"{m:+.3f} +/- {s:.3f} (n={n})"


def _render_ablation_table_md(
    grid: Dict[Tuple[str, str, str], Dict[str, float]],
    protocol: str,
) -> List[str]:
    """Render a 4-method x 4-ablation Sharpe grid for one protocol."""
    header = "| method \\ ablation | " + " | ".join(_ABLATIONS) + " |"
    sep = "|" + "---|" * (1 + len(_ABLATIONS))
    lines: List[str] = [header, sep]
    for method in _METHODS:
        row_cells = [method]
        for abl in _ABLATIONS:
            key = (abl, method, protocol)
            entry = grid.get(key)
            if entry is None or entry["n"] == 0:
                # stripped_l3 + constant_full is undefined, show n/a.
                if abl == "stripped_l3" and method == "constant_full":
                    row_cells.append("n/a (undefined)")
                else:
                    row_cells.append("-- (n=0)")
                continue
            row_cells.append(_format_cell(
                entry["mean"], entry["std"], int(entry["n"]),
            ))
        lines.append("| " + " | ".join(row_cells) + " |")
    return lines


def _render_pooled_ds_table_md(
    grid: Dict[Tuple[str, str, str], Dict[str, float]],
    protocol: str,
) -> List[str]:
    """Render the day-stream pooled Sharpe table (secondary)."""
    header = "| method \\ ablation | " + " | ".join(_ABLATIONS) + " |"
    sep = "|" + "---|" * (1 + len(_ABLATIONS))
    lines: List[str] = [header, sep]
    for method in _METHODS:
        row_cells = [method]
        for abl in _ABLATIONS:
            key = (abl, method, protocol)
            entry = grid.get(key)
            if entry is None or entry["n"] == 0:
                if abl == "stripped_l3" and method == "constant_full":
                    row_cells.append("n/a")
                else:
                    row_cells.append("--")
                continue
            row_cells.append(f"{entry['ds_pooled']:+.3f}")
        lines.append("| " + " | ".join(row_cells) + " |")
    return lines


def _render_per_fold_appendix(
    cells_by_tuple: Dict[Tuple[str, str, str], List[Dict[str, object]]],
    protocol: str,
) -> List[str]:
    """Per-fold Sharpe drill-down for QA. Long; emitted in an appendix."""
    lines: List[str] = []
    for abl in _ABLATIONS:
        for method in _METHODS:
            if abl == "stripped_l3" and method == "constant_full":
                continue
            cells = cells_by_tuple.get((abl, method, protocol), [])
            if not cells:
                continue
            lines.append(f"### {abl} / {method} ({protocol})")
            lines.append("")
            lines.append("| fold | n_seeds |   mean |    std |")
            lines.append("|-----:|--------:|-------:|-------:|")
            for f, n, m, s in _per_fold_means(cells):
                lines.append(f"| {f:>4} | {n:>7} | {m:+.4f} | {s:.4f} |")
            lines.append("")
    return lines


def _build_grid(
    root: Path,
) -> Tuple[
    Dict[Tuple[str, str, str], Dict[str, float]],
    Dict[Tuple[str, str, str], List[Dict[str, object]]],
]:
    """Load every tuple's cells; return (summary grid, per-cell map)."""
    grid: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    cells_by_tuple: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    for abl in _ABLATIONS:
        for method in _METHODS:
            for protocol in _PROTOCOLS:
                if abl == "stripped_l3" and method == "constant_full":
                    continue
                cells = _load_tuple_cells(root, abl, method, protocol)
                cells_by_tuple[(abl, method, protocol)] = cells
                m, s, n = _per_cell_mean_sharpe(cells)
                ds = _day_stream_pooled_sharpe(cells)
                grid[(abl, method, protocol)] = {
                    "mean": m, "std": s, "n": n, "ds_pooled": ds,
                }
    return grid, cells_by_tuple


def _render_markdown(
    grid: Dict[Tuple[str, str, str], Dict[str, float]],
    cells_by_tuple: Dict[Tuple[str, str, str], List[Dict[str, object]]],
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# NASDAQ-100 Phase 6: Four-Ablation Replication")
    lines.append("")
    lines.append(
        "Cross-universe transferability test for the InVAR-RL mechanism "
        "claim. The Phase 6 four-ablation table on the NASDAQ-100 panel "
        "mirrors the SP500 Phase 6 Table 4 byte-for-byte (same 5 macro-"
        "stratified folds, same 5 seeds 42-46, same val window, same "
        "SAC/PPO/RecurrentPPO architectures, same Ledoit-Wolf 120-day "
        "covariance + cvxpy QP at gamma=5, per-name cap 0.05, gross=1)."
    )
    lines.append("")
    lines.append("## Ablation conditions")
    lines.append("")
    lines.append(
        "- **canonical**: full InVAR-RL stack (Layer 1 InVAR ranker + "
        "Layer 2 Ledoit-Wolf MV-QP + Layer 3 RL controller)."
    )
    lines.append(
        "- **random_l1**: Layer 1 InVAR scores replaced with i.i.d. "
        "N(0, 1) seeded by cell seed. Tests whether Sharpe comes from "
        "the InVAR ranker or from QP + RL exposure logic on any signal."
    )
    lines.append(
        "- **equal_l2 (K=20 per side, NDX-100)**: Layer 2 QP replaced "
        "with naive equal-weight top-20 long / bottom-20 short. The "
        "NDX-100 K is 20 vs SP500's 50 because the active universe is "
        "roughly 1/5 the size, so 20/100 matches the SP500 50/250 "
        "fraction. Tests how much of the lower-stack value is the QP."
    )
    lines.append(
        "- **stripped_l3**: Layer 1 + Layer 2 canonical, but the "
        "Layer-1 / Layer-2 fields of the RL observation "
        "(score_dispersion, pred_vol, eff_positions, full macro "
        "encoding) are zeroed via StrippedObservationWrapper. Risk-"
        "state fields (rolling vol, drawdown, exposure, days since "
        "regime change) are preserved. constant_full is unaffected "
        "(no observation read), so the entry is n/a."
    )
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append("- **recurrent_ppo**: sb3-contrib RecurrentPPO (MlpLstmPolicy).")
    lines.append("- **feedforward_ppo**: SB3 PPO (MlpPolicy).")
    lines.append("- **sac**: SB3 SAC (MlpPolicy).")
    lines.append(
        "- **constant_full**: holds the L1 + L2 portfolio with exposure "
        "1.0 (no Layer 3 intervention). No training."
    )
    lines.append("")
    lines.append("## Pooling formula")
    lines.append("")
    lines.append(
        "PRIMARY = per-cell annualised Sharpe (sqrt(252) scaling) "
        "averaged across the 25 (fold, seed) cells of each tuple "
        "(matches Phase 5 + Phase 5.5 rollup). Secondary = day-stream "
        "pooled Sharpe (every cell's daily strategy return "
        "concatenated, then one Sharpe)."
    )
    lines.append("")

    for protocol in _PROTOCOLS:
        label = "Long-short (L/S)" if protocol == "ls" else "Long-only (L/O)"
        lines.append(f"## {label}: per-cell mean Sharpe (PRIMARY)")
        lines.append("")
        lines.extend(_render_ablation_table_md(grid, protocol))
        lines.append("")
        lines.append(f"## {label}: day-stream pooled Sharpe (secondary)")
        lines.append("")
        lines.extend(_render_pooled_ds_table_md(grid, protocol))
        lines.append("")

    lines.append("## Per-fold drill-down (appendix)")
    lines.append("")
    for protocol in _PROTOCOLS:
        label = "Long-short (L/S)" if protocol == "ls" else "Long-only (L/O)"
        lines.append(f"### {label}")
        lines.append("")
        lines.extend(
            _render_per_fold_appendix(cells_by_tuple, protocol)
        )

    lines.append("## Cross-universe comparison vs SP500 Phase 6 Table 4")
    lines.append("")
    lines.append(
        "TODO: fill in the side-by-side once both grids are complete. "
        "Replication is graded along three axes:"
    )
    lines.append("")
    lines.append(
        "1. **random_l1 collapse**: SP500 canonical SAC Sharpe should "
        "collapse to near 0 under random_l1. NDX-100 should show the "
        "same direction (positive canonical lift, near-zero under "
        "random L1); a sign-preserving large absolute Sharpe under "
        "random_l1 would falsify the Layer 1 mechanism."
    )
    lines.append(
        "2. **equal_l2 partial-recovery**: SP500 equal_l2 preserves "
        "some Sharpe (the L1 score still ranks; only the QP risk "
        "model is gone). NDX-100 should preserve a comparable "
        "fraction, modulo universe-size effects (K=20 NDX vs K=50 "
        "SP500)."
    )
    lines.append(
        "3. **stripped_l3 collapse**: SP500 stripped_l3 collapses for "
        "the RL methods (recurrent_ppo, feedforward_ppo, sac) since "
        "the agent can no longer condition on the L1/L2 regime "
        "signal. NDX-100 should show the same collapse pattern."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> int:
    root = Path("outputs/nasdaq100/phase6_ablation")
    grid, cells_by_tuple = _build_grid(root)
    out_path = Path("reports/nasdaq100/phase_6_ablations.md")
    _render_markdown(grid, cells_by_tuple, out_path)
    # Coverage report on stdout.
    total = 0
    for (abl, method, protocol), entry in grid.items():
        total += int(entry["n"])
    print(
        f"[rollup phase6] wrote {out_path}: total cells loaded={total} "
        f"(target = 30 tuples x 25 cells = 750 worst case; "
        f"canonical/sac/ls + canonical/sac/lo overlap Phase 5)"
    )
    for protocol in _PROTOCOLS:
        label = "L/S" if protocol == "ls" else "L/O"
        print(f"  [{label}]")
        for abl in _ABLATIONS:
            for method in _METHODS:
                if abl == "stripped_l3" and method == "constant_full":
                    continue
                entry = grid.get((abl, method, protocol))
                if entry is None:
                    continue
                m, s, n = entry["mean"], entry["std"], entry["n"]
                ds = entry["ds_pooled"]
                print(
                    f"    {abl:13s} {method:18s} n={n:>2} "
                    f"per-cell={m:+.4f} +/- {s:.4f}  "
                    f"day-stream={ds:+.4f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
