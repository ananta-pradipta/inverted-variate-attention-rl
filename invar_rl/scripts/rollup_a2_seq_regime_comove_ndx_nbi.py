"""Roll up the A2 sequential regime->co-movement NDX + NBI sweep.

Reads L2L3 summary JSONs under
``outputs/nasdaq100/layer3_a2_seq_regime_comove/ls/summary/`` and
``outputs/biotech_nbi_enriched/layer3_a2_seq_regime_comove_k25/ls/summary/``
and emits per-universe stats compared to the corresponding canonical
SAC L/S reference (matched to the C3 cross-universe report references).

Canonical references (audited per bcbabc2):
    NDX-100               pool +1.194  (F1 +1.97, F2 -0.34, F3 +0.72,
                                         F4 +1.74, F5 +1.87)
    biotech_nbi_enriched  pool +1.541  (F1 +2.958, F2 +1.719, F3 +1.314,
                                         F4 +0.875, F5 +0.840)

Stop gates (from the task spec):
    NDX:
        Pool >= +1.144 (FLOOR / WIN if >= +1.244 / STRONG WIN if >= +1.294)
    NBI:
        Pool >= +1.491 (FLOOR / WIN if >= +1.591 / STRONG WIN if >= +1.641)
        F4 + F5 sum >= canonical (+1.715) - 0.20 = +1.515
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
        1: +2.958, 2: +1.719, 3: +1.314, 4: +0.875, 5: +0.840,
    },
}

DEFAULT_LS_DIR: Dict[str, Path] = {
    "nasdaq100": Path(
        "outputs/nasdaq100/layer3_a2_seq_regime_comove/ls"
    ),
    "biotech_nbi_enriched": Path(
        "outputs/biotech_nbi_enriched/layer3_a2_seq_regime_comove_k25/ls"
    ),
}

STOP_GATES: Dict[str, Dict[str, float]] = {
    "nasdaq100": {
        "floor": +1.144,
        "win": +1.244,
        "strong_win": +1.294,
    },
    "biotech_nbi_enriched": {
        "floor": +1.491,
        "win": +1.591,
        "strong_win": +1.641,
        "f45_floor": +1.515,
    },
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


def roll_universe(
    universe: str, ls_root: Path,
) -> Dict[str, object]:
    per_fold: Dict[int, Dict[str, object]] = {}
    pool_cells: List[float] = []
    n_present = 0
    n_total = 0
    for f in FOLDS:
        cells: List[float] = []
        for s in SEEDS:
            v = _load_cell(ls_root, f, s)
            n_total += 1
            if v == v:
                n_present += 1
            cells.append(v)
            pool_cells.append(v)
        m, sd = _mean_sd(cells)
        per_fold[f] = {
            "mean": m, "sd": sd, "cells": cells,
            "canonical": CANONICAL_PER_FOLD[universe][f],
            "delta": (m - CANONICAL_PER_FOLD[universe][f]) if m == m else float("nan"),
        }
    pool_mean, pool_sd = _mean_sd(pool_cells)
    sos = _sos(pool_cells)
    canon_pool = CANONICAL_POOL[universe]
    delta = (pool_mean - canon_pool) if pool_mean == pool_mean else float("nan")

    gates = STOP_GATES[universe]
    verdict_flags: List[str] = []
    if pool_mean == pool_mean:
        if pool_mean >= gates.get("strong_win", float("inf")):
            verdict_flags.append("STRONG WIN")
        elif pool_mean >= gates.get("win", float("inf")):
            verdict_flags.append("WIN")
        elif pool_mean >= gates.get("floor", float("inf")):
            verdict_flags.append("PASS (floor)")
        else:
            verdict_flags.append("FAIL (below floor)")
    else:
        verdict_flags.append("INCOMPLETE")
    if universe == "biotech_nbi_enriched":
        f45 = per_fold[4]["mean"] + per_fold[5]["mean"] if (
            per_fold[4]["mean"] == per_fold[4]["mean"]
            and per_fold[5]["mean"] == per_fold[5]["mean"]
        ) else float("nan")
        if f45 == f45:
            if f45 >= gates["f45_floor"]:
                verdict_flags.append(f"F4+F5 OK ({f45:+.3f} >= {gates['f45_floor']:+.3f})")
            else:
                verdict_flags.append(f"F4+F5 BELOW FLOOR ({f45:+.3f} < {gates['f45_floor']:+.3f})")
    return {
        "universe": universe,
        "n_present": n_present, "n_total": n_total,
        "pool_mean": pool_mean, "pool_sd": pool_sd, "pool_sos": sos,
        "canonical_pool": canon_pool, "delta": delta,
        "per_fold": per_fold,
        "verdict": "; ".join(verdict_flags),
        "ls_root": str(ls_root),
    }


def _fmt_pool(d: Dict[str, object]) -> str:
    pm = d["pool_mean"]
    sd = d["pool_sd"]
    sos = d["pool_sos"]
    delta = d["delta"]
    cp = d["canonical_pool"]
    nc = d["n_present"]
    nt = d["n_total"]
    if pm != pm:
        return (
            f"[{d['universe']}] cells={nc}/{nt} pool=INCOMPLETE "
            f"canonical={cp:+.3f} verdict={d['verdict']}"
        )
    return (
        f"[{d['universe']}] cells={nc}/{nt} "
        f"pool={pm:+.3f} sd={sd:.3f} SoS={sos:.3f} "
        f"canonical={cp:+.3f} delta={delta:+.3f} "
        f"verdict={d['verdict']}"
    )


def _fmt_per_fold_table(d: Dict[str, object]) -> str:
    rows = []
    rows.append("| Fold | canonical | A2 seq | delta |")
    rows.append("|---:|---:|---:|---:|")
    for f in FOLDS:
        pf = d["per_fold"][f]
        m = pf["mean"]
        c = pf["canonical"]
        de = pf["delta"]
        if m == m:
            rows.append(f"| F{f} | {c:+.3f} | {m:+.3f} | {de:+.3f} |")
        else:
            rows.append(f"| F{f} | {c:+.3f} | INCOMPLETE | n/a |")
    pm = d["pool_mean"]
    cp = d["canonical_pool"]
    de = d["delta"]
    if pm == pm:
        rows.append(f"| **Pool** | **{cp:+.3f}** | **{pm:+.3f}** | **{de:+.3f}** |")
    else:
        rows.append(f"| **Pool** | **{cp:+.3f}** | **INCOMPLETE** | n/a |")
    return "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Roll up A2 sequential regime->co-movement NDX + NBI sweep."
        )
    )
    p.add_argument(
        "--ndx-root", type=Path,
        default=DEFAULT_LS_DIR["nasdaq100"],
    )
    p.add_argument(
        "--nbi-root", type=Path,
        default=DEFAULT_LS_DIR["biotech_nbi_enriched"],
    )
    p.add_argument(
        "--out-json", type=Path,
        default=Path(
            "reports/pretrain_improvements/"
            "a2_seq_regime_comove_ndx_nbi_2026-05-27.json"
        ),
    )
    args = p.parse_args()

    ndx = roll_universe("nasdaq100", args.ndx_root)
    nbi = roll_universe("biotech_nbi_enriched", args.nbi_root)

    print(_fmt_pool(ndx))
    print()
    print(_fmt_per_fold_table(ndx))
    print()
    print(_fmt_pool(nbi))
    print()
    print(_fmt_per_fold_table(nbi))

    payload = {
        "nasdaq100": {k: v for k, v in ndx.items() if k != "per_fold"} | {
            "per_fold": {
                str(f): {
                    "mean": ndx["per_fold"][f]["mean"],
                    "sd": ndx["per_fold"][f]["sd"],
                    "canonical": ndx["per_fold"][f]["canonical"],
                    "delta": ndx["per_fold"][f]["delta"],
                } for f in FOLDS
            }
        },
        "biotech_nbi_enriched": {
            k: v for k, v in nbi.items() if k != "per_fold"
        } | {
            "per_fold": {
                str(f): {
                    "mean": nbi["per_fold"][f]["mean"],
                    "sd": nbi["per_fold"][f]["sd"],
                    "canonical": nbi["per_fold"][f]["canonical"],
                    "delta": nbi["per_fold"][f]["delta"],
                } for f in FOLDS
            }
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
