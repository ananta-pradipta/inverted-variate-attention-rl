"""Roll up the InVAR-RL-SIA Phase 4c cross-universe ablation matrix.

For each of three universes (SP500, NASDAQ-100, biotech-NBI-enriched),
load the four SIA variants (full, no_a, no_s, no_i) from per-cell summary
JSONs (5 folds x 5 seeds = 25 cells each). Compute pooled mean, pooled sd,
per-fold means, and the per-component lift attribution

    A_lift = full - no_a
    S_lift = full - no_s
    I_lift = full - no_i

across pooled mean, F2 mean, and pooled sd. Compare against the canonical
SAC baseline per universe.

Outputs:

  - Per-universe variant table (canonical, full, no_a, no_s, no_i)
  - 3-universe x 3-component lift matrix (pooled / F2 / sd / Sharpe-of-Sharpes)
  - JSON dump for downstream consumption.

Usage::

    python -m invar_rl.scripts.rollup_sia_phase4c_cross_universe \
        --json-out reports/sia/_phase_4c_cross_universe_rollup.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple


_FOLDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)

_UNIVERSES = ("sp500", "nasdaq100", "biotech_nbi_enriched")

# Per-universe layout: full_sia and three ablations live at known dirs.
# full_sia and no_a/no_s/no_i are produced by the Phase 4/4b/4c sbatches.
_LAYOUT: Dict[str, Dict[str, str]] = {
    "sp500": {
        "full_sia": "outputs/sp500/layer2_sia/phase2_regime_beta_1e-4",
        "no_a": "outputs/sp500/layer2_sia/phase4_no_a",
        "no_s": "outputs/sp500/layer2_sia/phase4_no_s",
        "no_i": "outputs/sp500/layer2_sia/phase4_no_i",
        "canonical_pooled": 0.945,
        "canonical_f2": -0.229,
        "canonical_sd": 0.977,
    },
    "nasdaq100": {
        "full_sia": "outputs/nasdaq100/layer2_sia/phase3",
        "no_a": "outputs/nasdaq100/layer2_sia/phase4c_no_a",
        "no_s": "outputs/nasdaq100/layer2_sia/phase4c_no_s",
        "no_i": "outputs/nasdaq100/layer2_sia/phase4c_no_i",
        "canonical_pooled": 1.194,
        "canonical_f2": -0.34,
        "canonical_sd": 1.215,
    },
    "biotech_nbi_enriched": {
        "full_sia": "outputs/biotech_nbi_enriched/layer2_sia/phase3",
        "no_a": "outputs/biotech_nbi_enriched/layer2_sia/phase4c_no_a",
        "no_s": "outputs/biotech_nbi_enriched/layer2_sia/phase4c_no_s",
        "no_i": "outputs/biotech_nbi_enriched/layer2_sia/phase4c_no_i",
        "canonical_pooled": 1.541,
        "canonical_f2": 1.91,
        "canonical_sd": 1.27,
    },
}


def _safe_sd(xs: List[float]) -> float:
    return float(pstdev(xs)) if len(xs) > 1 else 0.0


def _load_cell(summary_dir: Path, fold: int, seed: int) -> Optional[dict]:
    p = summary_dir / f"fold{fold}_seed{seed}.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def _load_variant_matrix(
    summary_dir: Path,
) -> Tuple[Dict[Tuple[int, int], Optional[float]], List[Tuple[int, int]]]:
    matrix: Dict[Tuple[int, int], Optional[float]] = {}
    missing: List[Tuple[int, int]] = []
    for f in _FOLDS:
        for s in _SEEDS:
            payload = _load_cell(summary_dir, f, s)
            if payload is None:
                matrix[(f, s)] = None
                missing.append((f, s))
            else:
                matrix[(f, s)] = float(payload["test_pooled_sharpe"])
    return matrix, missing


def _summarise_variant(
    matrix: Dict[Tuple[int, int], Optional[float]],
) -> Dict[str, object]:
    pooled_cells: List[float] = []
    fold_means: Dict[int, float] = {}
    fold_sds: Dict[int, float] = {}
    for f in _FOLDS:
        cells = [matrix[(f, s)] for s in _SEEDS]
        present = [c for c in cells if c is not None]
        pooled_cells.extend(present)
        fold_means[f] = mean(present) if present else float("nan")
        fold_sds[f] = _safe_sd(present)
    pooled_mean = mean(pooled_cells) if pooled_cells else float("nan")
    pooled_sd = _safe_sd(pooled_cells)
    pooled_sem = (
        pooled_sd / math.sqrt(len(pooled_cells)) if pooled_cells else 0.0
    )
    sos = (pooled_mean / pooled_sd) if pooled_sd > 0 else float("nan")
    return {
        "n_cells": len(pooled_cells),
        "pooled_mean": pooled_mean,
        "pooled_sd": pooled_sd,
        "pooled_sem": pooled_sem,
        "sharpe_of_sharpes": sos,
        "per_fold_mean": {str(f): fold_means[f] for f in _FOLDS},
        "per_fold_sd": {str(f): fold_sds[f] for f in _FOLDS},
        "per_cell": {
            f"fold{f}_seed{s}": matrix[(f, s)]
            for f in _FOLDS for s in _SEEDS
        },
    }


def _load_universe(uni: str) -> Dict[str, object]:
    layout = _LAYOUT[uni]
    summaries: Dict[str, object] = {}
    missing_counts: Dict[str, int] = {}
    for variant in ("full_sia", "no_a", "no_s", "no_i"):
        root = Path(str(layout[variant]))
        sdir = root / "summary"
        if not sdir.exists():
            summaries[variant] = None
            missing_counts[variant] = 25
            continue
        matrix, missing = _load_variant_matrix(sdir)
        summaries[variant] = _summarise_variant(matrix)
        missing_counts[variant] = len(missing)
    return {
        "universe": uni,
        "summaries": summaries,
        "missing_counts": missing_counts,
        "canonical_pooled": float(layout["canonical_pooled"]),
        "canonical_f2": float(layout["canonical_f2"]),
        "canonical_sd": float(layout["canonical_sd"]),
    }


def _component_lifts(
    summaries: Dict[str, object],
) -> Dict[str, Dict[str, float]]:
    """Compute per-component lift on pooled / F2 / sd / Sharpe-of-Sharpes."""

    out: Dict[str, Dict[str, float]] = {}
    full = summaries.get("full_sia")
    if full is None:
        return out
    full_pool = float(full["pooled_mean"])
    full_f2 = float(full["per_fold_mean"]["2"])
    full_sd = float(full["pooled_sd"])
    full_sos = float(full["sharpe_of_sharpes"]) if full["sharpe_of_sharpes"] == full["sharpe_of_sharpes"] else float("nan")
    for comp, key in (("A", "no_a"), ("S", "no_s"), ("I", "no_i")):
        ablated = summaries.get(key)
        if ablated is None:
            out[comp] = {
                "pooled_lift": float("nan"),
                "f2_lift": float("nan"),
                "sd_reduction": float("nan"),
                "sos_lift": float("nan"),
            }
            continue
        ab_pool = float(ablated["pooled_mean"])
        ab_f2 = float(ablated["per_fold_mean"]["2"])
        ab_sd = float(ablated["pooled_sd"])
        ab_sos = float(ablated["sharpe_of_sharpes"]) if ablated["sharpe_of_sharpes"] == ablated["sharpe_of_sharpes"] else float("nan")
        out[comp] = {
            # lift = full - ablated (positive means component HELPS that metric)
            "pooled_lift": full_pool - ab_pool,
            "f2_lift": full_f2 - ab_f2,
            # sd_reduction = ablated_sd - full_sd (positive means component LOWERS variance)
            "sd_reduction": ab_sd - full_sd,
            "sos_lift": full_sos - ab_sos,
        }
    return out


def _print_universe_table(uni_data: Dict[str, object]) -> None:
    uni = uni_data["universe"]
    summaries = uni_data["summaries"]
    print(f"## {uni}")
    print()
    print(
        "| variant | n | pooled | F1 | F2 | F3 | F4 | F5 | sd | SoS |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    canon_pool = float(uni_data["canonical_pooled"])
    canon_f2 = float(uni_data["canonical_f2"])
    canon_sd = float(uni_data["canonical_sd"])
    canon_sos = (canon_pool / canon_sd) if canon_sd > 0 else float("nan")
    print(
        f"| canonical SAC | NA | {canon_pool:+.3f} | NA | {canon_f2:+.3f} | "
        f"NA | NA | NA | {canon_sd:.3f} | {canon_sos:+.3f} |"
    )
    for variant in ("full_sia", "no_a", "no_s", "no_i"):
        s = summaries.get(variant)
        if s is None:
            print(f"| {variant} | 0 | MISS | MISS | MISS | MISS | MISS | MISS | MISS | MISS |")
            continue
        row = (
            f"| {variant} | {int(s['n_cells'])} | {float(s['pooled_mean']):+.3f} | "
            f"{float(s['per_fold_mean']['1']):+.3f} | "
            f"{float(s['per_fold_mean']['2']):+.3f} | "
            f"{float(s['per_fold_mean']['3']):+.3f} | "
            f"{float(s['per_fold_mean']['4']):+.3f} | "
            f"{float(s['per_fold_mean']['5']):+.3f} | "
            f"{float(s['pooled_sd']):.3f} | "
            f"{float(s['sharpe_of_sharpes']):+.3f} |"
        )
        print(row)
    print()


def _print_lift_matrix(
    universes_data: Dict[str, Dict[str, object]],
) -> None:
    lifts_by_uni: Dict[str, Dict[str, Dict[str, float]]] = {}
    for uni, data in universes_data.items():
        lifts_by_uni[uni] = _component_lifts(data["summaries"])

    print("## Cross-universe per-component lift matrix")
    print()
    print("Pooled-mean lift (full minus ablated; positive = component helps):")
    print()
    print("| Component | SP500 | NDX | NBI |")
    print("|---|---:|---:|---:|")
    for comp in ("A", "S", "I"):
        sp = lifts_by_uni["sp500"].get(comp, {}).get("pooled_lift", float("nan"))
        nd = lifts_by_uni["nasdaq100"].get(comp, {}).get("pooled_lift", float("nan"))
        nb = lifts_by_uni["biotech_nbi_enriched"].get(comp, {}).get("pooled_lift", float("nan"))
        print(f"| {comp} | {sp:+.3f} | {nd:+.3f} | {nb:+.3f} |")
    print()

    print("F2 lift (full F2 minus ablated F2):")
    print()
    print("| Component | SP500 | NDX | NBI |")
    print("|---|---:|---:|---:|")
    for comp in ("A", "S", "I"):
        sp = lifts_by_uni["sp500"].get(comp, {}).get("f2_lift", float("nan"))
        nd = lifts_by_uni["nasdaq100"].get(comp, {}).get("f2_lift", float("nan"))
        nb = lifts_by_uni["biotech_nbi_enriched"].get(comp, {}).get("f2_lift", float("nan"))
        print(f"| {comp} | {sp:+.3f} | {nd:+.3f} | {nb:+.3f} |")
    print()

    print("Sd reduction (ablated sd minus full sd; positive = component lowers variance):")
    print()
    print("| Component | SP500 | NDX | NBI |")
    print("|---|---:|---:|---:|")
    for comp in ("A", "S", "I"):
        sp = lifts_by_uni["sp500"].get(comp, {}).get("sd_reduction", float("nan"))
        nd = lifts_by_uni["nasdaq100"].get(comp, {}).get("sd_reduction", float("nan"))
        nb = lifts_by_uni["biotech_nbi_enriched"].get(comp, {}).get("sd_reduction", float("nan"))
        print(f"| {comp} | {sp:+.3f} | {nd:+.3f} | {nb:+.3f} |")
    print()

    print("Sharpe-of-Sharpes lift (full SoS minus ablated SoS):")
    print()
    print("| Component | SP500 | NDX | NBI |")
    print("|---|---:|---:|---:|")
    for comp in ("A", "S", "I"):
        sp = lifts_by_uni["sp500"].get(comp, {}).get("sos_lift", float("nan"))
        nd = lifts_by_uni["nasdaq100"].get(comp, {}).get("sos_lift", float("nan"))
        nb = lifts_by_uni["biotech_nbi_enriched"].get(comp, {}).get("sos_lift", float("nan"))
        print(f"| {comp} | {sp:+.3f} | {nd:+.3f} | {nb:+.3f} |")
    print()
    return lifts_by_uni


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json-out", type=Path,
        default=Path("reports/sia/_phase_4c_cross_universe_rollup.json"),
    )
    args = ap.parse_args()

    print("# InVAR-RL-SIA Phase 4c cross-universe ablation rollup")
    print()

    universes_data: Dict[str, Dict[str, object]] = {}
    for uni in _UNIVERSES:
        universes_data[uni] = _load_universe(uni)

    for uni in _UNIVERSES:
        _print_universe_table(universes_data[uni])

    lifts = _print_lift_matrix(universes_data)

    payload = {
        "universes": {},
        "lifts": lifts,
    }
    for uni, data in universes_data.items():
        payload["universes"][uni] = {
            "canonical_pooled": data["canonical_pooled"],
            "canonical_f2": data["canonical_f2"],
            "canonical_sd": data["canonical_sd"],
            "summaries": data["summaries"],
            "missing_counts": data["missing_counts"],
        }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
