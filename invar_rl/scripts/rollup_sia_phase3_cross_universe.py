"""Roll up the SIA Phase 3 cross-universe sweep into a 25-cell summary.

For each of NDX-100 and NBI-enriched, reads
``outputs/{universe}/layer2_sia/phase3/summary/fold{F}_seed{S}.json``
for 5 folds x 5 seeds and emits:
- per-cell test pooled Sharpe matrix
- per-fold mean + sd
- pooled 25-cell mean + sd
- delta vs canonical SAC L/S (hardcoded references):
    - NDX-100 K=20: +1.194
    - NBI-enriched K=25: +1.541
- verdict per universe vs predicted lift bands:
    - NDX-100: +0.03 to +0.10 (PASS), 0 to +0.03 (PARTIAL), <0 (FAIL)
    - NBI-enriched: +0.10 to +0.25 (PASS), 0 to +0.10 (PARTIAL), <0 (FAIL)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


SEEDS = (42, 43, 44, 45, 46)
FOLDS = (1, 2, 3, 4, 5)

CANONICAL_SAC: Dict[str, float] = {
    "nasdaq100": +1.194,
    "biotech_nbi_enriched": +1.541,
}

# Predicted lift bands (low, high) above canonical SAC.
PREDICTED_BANDS: Dict[str, Tuple[float, float]] = {
    "nasdaq100": (0.03, 0.10),
    "biotech_nbi_enriched": (0.10, 0.25),
}


def _load_cell(root: Path, fold: int, seed: int) -> float:
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


def _verdict(universe: str, pooled_mean: float) -> str:
    canon = CANONICAL_SAC[universe]
    lo, hi = PREDICTED_BANDS[universe]
    delta = pooled_mean - canon
    if pooled_mean != pooled_mean:
        return "INCOMPLETE"
    if delta >= lo:
        if delta >= hi:
            return f"PASS_STRONG (delta=+{delta:.4f} >= upper band +{hi:.2f})"
        return f"PASS (delta=+{delta:.4f} in band +{lo:.2f} to +{hi:.2f})"
    if delta > 0:
        return f"PARTIAL (delta=+{delta:.4f}, below low band +{lo:.2f})"
    return f"FAIL (delta={delta:+.4f} below canonical)"


def _rollup_universe(universe: str, root: Path) -> Dict[str, object]:
    matrix: Dict[int, Dict[int, float]] = {}
    all_cells: List[float] = []
    per_fold_means: Dict[int, float] = {}
    per_fold_sds: Dict[int, float] = {}
    n_complete = 0
    for f in FOLDS:
        row: Dict[int, float] = {}
        sharpes: List[float] = []
        for s in SEEDS:
            sh = _load_cell(root, f, s)
            row[s] = sh
            if sh == sh:
                sharpes.append(sh)
                all_cells.append(sh)
                n_complete += 1
        matrix[f] = row
        m, sd = _mean_sd(sharpes)
        per_fold_means[f] = m
        per_fold_sds[f] = sd
    pooled_mean, pooled_sd = _mean_sd(all_cells)
    canon = CANONICAL_SAC[universe]
    return {
        "universe": universe,
        "root": str(root),
        "n_complete": n_complete,
        "matrix": {f: matrix[f] for f in FOLDS},
        "per_fold_mean": per_fold_means,
        "per_fold_sd": per_fold_sds,
        "pooled_mean": pooled_mean,
        "pooled_sd": pooled_sd,
        "canonical_sac_pooled": canon,
        "delta_vs_canonical": pooled_mean - canon,
        "predicted_band": PREDICTED_BANDS[universe],
        "verdict": _verdict(universe, pooled_mean),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ndx-root",
        default="outputs/nasdaq100/layer2_sia/phase3",
    )
    p.add_argument(
        "--nbi-root",
        default="outputs/biotech_nbi_enriched/layer2_sia/phase3",
    )
    p.add_argument(
        "--out", default="reports/sia/_phase_3_cross_universe_rollup.json"
    )
    args = p.parse_args()

    ndx = _rollup_universe("nasdaq100", Path(args.ndx_root))
    nbi = _rollup_universe("biotech_nbi_enriched", Path(args.nbi_root))

    rollup = {
        "nasdaq100": ndx,
        "biotech_nbi_enriched": nbi,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(rollup, fh, indent=2)
    print(json.dumps(rollup, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
