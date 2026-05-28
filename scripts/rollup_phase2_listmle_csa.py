"""Rollup script for InVAR-RL F2 (ListMLE) + F3 (CSA) Phase 2 SP500 25-cell sweep.

Reads JSONs from:
  - outputs/sp500/layer3_f2_listmle_phase2/sac_ls/foldF_seedS.json
  - outputs/sp500/layer3_f2_listmle_phase2/wrapper/invar_l1/foldF_seedS.json
  - outputs/sp500/layer3_f3_csa_phase2/sac_ls/foldF_seedS.json
  - outputs/sp500/layer3_f3_csa_phase2/wrapper/invar_l1/foldF_seedS.json
  - outputs/sp500/layer3_k25/ls/foldF_seedS.json (canonical reference)

Computes per-cell SAC Sharpe (annualised), per-fold means, pooled 25-cell
mean + sd + Sharpe-of-Sharpes. Emits a markdown table for the report.

Usage:
    python scripts/rollup_phase2_listmle_csa.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SQRT_252 = math.sqrt(252.0)


def _sharpe(mean_ret: float, vol: float) -> float:
    if vol is None or vol <= 1e-12:
        return 0.0
    return (mean_ret / vol) * SQRT_252


def load_sac_cell(path: Path) -> Optional[float]:
    """Return annualised Sharpe for a stage3 SAC cell JSON, or None if missing."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    sac = d.get("methods", {}).get("sac")
    if sac is None:
        return None
    mr = float(sac.get("mean_return", 0.0))
    vol = float(sac.get("volatility", 0.0))
    return _sharpe(mr, vol)


def load_wrapper_cell(path: Path) -> Optional[float]:
    """Return L/S Sharpe for a wrapper invar_l1 cell JSON, or None if missing."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    s = d.get("sharpe_ls")
    if s is None:
        # Fallback: dig into methods.
        methods = d.get("methods", {})
        for k, v in methods.items():
            if "long_short" in k:
                return float(v.get("sharpe_annualised", 0.0))
        return None
    return float(s)


def collect_25cells(
    root: Path,
    kind: str,
) -> Dict[Tuple[int, int], Optional[float]]:
    """Collect all (fold, seed) Sharpes from a root dir.

    kind == "sac"     : reads {root}/foldF_seedS.json (stage3_rl_ablation output)
    kind == "wrapper" : reads {root}/invar_l1/foldF_seedS.json
    """
    out: Dict[Tuple[int, int], Optional[float]] = {}
    for fold in (1, 2, 3, 4, 5):
        for seed in (42, 43, 44, 45, 46):
            if kind == "sac":
                p = root / f"fold{fold}_seed{seed}.json"
                out[(fold, seed)] = load_sac_cell(p)
            elif kind == "wrapper":
                p = root / "invar_l1" / f"fold{fold}_seed{seed}.json"
                out[(fold, seed)] = load_wrapper_cell(p)
            else:
                raise ValueError(kind)
    return out


def pool_stats(cells: Dict[Tuple[int, int], Optional[float]]) -> Dict[str, float]:
    """Compute pooled mean + sd + Sharpe-of-Sharpes across all 25 cells."""
    vals = [v for v in cells.values() if v is not None]
    if not vals:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "sos": 0.0}
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(n - 1, 1))
    sos = mean / sd if sd > 1e-12 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "sos": sos}


def per_fold_means(
    cells: Dict[Tuple[int, int], Optional[float]],
) -> Dict[int, Optional[float]]:
    out: Dict[int, Optional[float]] = {}
    for fold in (1, 2, 3, 4, 5):
        vals = [
            cells[(fold, s)] for s in (42, 43, 44, 45, 46)
            if cells.get((fold, s)) is not None
        ]
        out[fold] = (sum(vals) / len(vals)) if vals else None
    return out


def fmt_cell(v: Optional[float]) -> str:
    return f"{v:+.3f}" if v is not None else "  n/a "


def emit_per_cell_table(
    cells_dict: Dict[str, Dict[Tuple[int, int], Optional[float]]],
    label: str,
) -> List[str]:
    """One section per variant, with a 5x5 grid (rows = fold, cols = seed)."""
    lines: List[str] = []
    lines.append(f"### {label}")
    lines.append("")
    lines.append("| Variant | F\\S | 42 | 43 | 44 | 45 | 46 | Fold mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for variant, cells in cells_dict.items():
        fold_means = per_fold_means(cells)
        for fold in (1, 2, 3, 4, 5):
            row = [
                fmt_cell(cells.get((fold, s)))
                for s in (42, 43, 44, 45, 46)
            ]
            fm = fold_means[fold]
            fm_str = f"{fm:+.3f}" if fm is not None else "  n/a "
            tag = variant if fold == 1 else ""
            lines.append(
                f"| {tag} | F{fold} | "
                + " | ".join(row)
                + f" | {fm_str} |"
            )
    lines.append("")
    return lines


def emit_pool_table(
    cells_dict: Dict[str, Dict[Tuple[int, int], Optional[float]]],
    label: str,
) -> List[str]:
    """Pool stats per variant in one row."""
    lines: List[str] = []
    lines.append(f"### {label}")
    lines.append("")
    lines.append("| Variant | n cells | Pool mean | Pool sd | Sharpe-of-Sharpes |")
    lines.append("|---|---:|---:|---:|---:|")
    for variant, cells in cells_dict.items():
        s = pool_stats(cells)
        lines.append(
            f"| {variant} | {int(s['n'])} | {s['mean']:+.3f} | "
            f"{s['sd']:.3f} | {s['sos']:+.3f} |"
        )
    lines.append("")
    return lines


def main() -> int:
    repo = Path(__file__).resolve().parent.parent

    # SAC cells.
    f2_sac = collect_25cells(
        repo / "outputs/sp500/layer3_f2_listmle_phase2/sac_ls", "sac"
    )
    f3_sac = collect_25cells(
        repo / "outputs/sp500/layer3_f3_csa_phase2/sac_ls", "sac"
    )
    can_sac = collect_25cells(
        repo / "outputs/sp500/layer3_k25/ls", "sac"
    )

    # Wrapper cells.
    f2_wrap = collect_25cells(
        repo / "outputs/sp500/layer3_f2_listmle_phase2/wrapper", "wrapper"
    )
    f3_wrap = collect_25cells(
        repo / "outputs/sp500/layer3_f3_csa_phase2/wrapper", "wrapper"
    )

    sac_dict = {
        "F2 (ListMLE)": f2_sac,
        "F3 (CSA)": f3_sac,
        "Canonical K=25": can_sac,
    }
    wrap_dict = {
        "F2 (ListMLE)": f2_wrap,
        "F3 (CSA)": f3_wrap,
    }

    out_lines: List[str] = []
    out_lines.append("# Phase 2 SP500 25-cell sweep: ListMLE (F2) + CSA (F3)")
    out_lines.append("")
    out_lines.append("Generated by `scripts/rollup_phase2_listmle_csa.py`.")
    out_lines.append("")
    out_lines.extend(emit_pool_table(sac_dict, "SAC L/S K=25 pooled stats"))
    out_lines.extend(emit_per_cell_table(sac_dict, "SAC L/S K=25 per-cell"))
    out_lines.extend(emit_pool_table(wrap_dict, "Wrapper L/S K=25 pooled stats"))
    out_lines.extend(emit_per_cell_table(wrap_dict, "Wrapper L/S K=25 per-cell"))

    print("\n".join(out_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
