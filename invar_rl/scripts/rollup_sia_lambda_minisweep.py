"""Roll up the SIA lambda_inv mini-sweep (Phase 2B) into a 3-way comparison.

Reads per-seed summary JSON for three lambda_inv values:
- 0.1 (Phase 2 reference): outputs/sp500/layer2_sia/phase2_regime_beta_1e-4/summary/fold1_seed{42..46}.json
- 0.3: outputs/sp500/layer2_sia/minisweep_lambda_inv_0.3/summary/fold1_seed{42..46}.json
- 1.0: outputs/sp500/layer2_sia/minisweep_lambda_inv_1.0/summary/fold1_seed{42..46}.json

Emits a 3-way table and picks the winner: the lambda_inv with the
highest F1 5-seed mean. Decision rule for Phase 3: if 0.3 or 1.0 beats
0.1 by >= +0.05, use the winner; else stick with 0.1.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


SEEDS = (42, 43, 44, 45, 46)


def _load_seed_sharpe(root: Path, seed: int) -> Tuple[float, float]:
    """Return (test_sharpe, aux_inv) for fold1 seed at root/summary/."""
    p = root / "summary" / f"fold1_seed{seed}.json"
    if not p.exists():
        return float("nan"), float("nan")
    with open(p) as fh:
        d = json.load(fh)
    sh = float(d.get("test_pooled_sharpe", float("nan")))
    inv = float(d.get("sia_train_stats", {}).get("aux_inv", float("nan")))
    return sh, inv


def _mean_sd(xs: List[float]) -> Tuple[float, float]:
    xs = [x for x in xs if x == x]  # drop NaN
    if not xs:
        return float("nan"), float("nan")
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ref-root",
        default="outputs/sp500/layer2_sia/phase2_regime_beta_1e-4",
        help="Phase 2 reference root (lambda_inv=0.1)",
    )
    p.add_argument(
        "--lambda-roots",
        nargs="+",
        default=[
            "outputs/sp500/layer2_sia/minisweep_lambda_inv_0.3",
            "outputs/sp500/layer2_sia/minisweep_lambda_inv_1.0",
        ],
    )
    p.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=[0.3, 1.0],
    )
    p.add_argument(
        "--out", default="reports/sia/_phase_2b_lambda_inv_sweep_rollup.json"
    )
    args = p.parse_args()

    rows: Dict[float, Dict[str, object]] = {}

    # Reference lambda_inv=0.1
    ref_sharpes, ref_invs = [], []
    for s in SEEDS:
        sh, inv = _load_seed_sharpe(Path(args.ref_root), s)
        ref_sharpes.append(sh)
        ref_invs.append(inv)
    ref_mean, ref_sd = _mean_sd(ref_sharpes)
    ref_inv_mean, _ = _mean_sd(ref_invs)
    rows[0.1] = {
        "lambda_inv": 0.1,
        "root": args.ref_root,
        "per_seed_sharpe": dict(zip(SEEDS, ref_sharpes)),
        "per_seed_aux_inv": dict(zip(SEEDS, ref_invs)),
        "f1_mean": ref_mean,
        "f1_sd": ref_sd,
        "aux_inv_mean": ref_inv_mean,
        "n_complete": sum(1 for x in ref_sharpes if x == x),
    }

    for root, lam in zip(args.lambda_roots, args.lambda_values):
        ss, invs = [], []
        for s in SEEDS:
            sh, inv = _load_seed_sharpe(Path(root), s)
            ss.append(sh)
            invs.append(inv)
        m, sd = _mean_sd(ss)
        inv_m, _ = _mean_sd(invs)
        rows[float(lam)] = {
            "lambda_inv": float(lam),
            "root": root,
            "per_seed_sharpe": dict(zip(SEEDS, ss)),
            "per_seed_aux_inv": dict(zip(SEEDS, invs)),
            "f1_mean": m,
            "f1_sd": sd,
            "aux_inv_mean": inv_m,
            "delta_vs_ref": m - ref_mean,
            "n_complete": sum(1 for x in ss if x == x),
        }

    # Decision rule
    candidates = [
        (lam, r["f1_mean"]) for lam, r in rows.items()
        if r.get("n_complete", 0) > 0 and r["f1_mean"] == r["f1_mean"]
    ]
    winner_lam = 0.1
    winner_mean = ref_mean
    for lam, m in candidates:
        if m > winner_mean:
            winner_mean = m
            winner_lam = lam
    use_for_phase3 = 0.1
    decision_note = ""
    if winner_lam != 0.1 and (winner_mean - ref_mean) >= 0.05:
        use_for_phase3 = winner_lam
        decision_note = (
            f"lambda_inv={winner_lam} beats 0.1 by "
            f"+{winner_mean - ref_mean:.4f} (>= +0.05 threshold); "
            f"using {winner_lam} for Phase 3."
        )
    else:
        decision_note = (
            f"no candidate beats 0.1 by >= +0.05 "
            f"(best alt delta = {winner_mean - ref_mean:+.4f}); "
            f"keeping lambda_inv=0.1 for Phase 3."
        )

    rollup = {
        "rows": {f"lambda_{k}": v for k, v in rows.items()},
        "winner_lambda_inv": winner_lam,
        "winner_f1_mean": winner_mean,
        "lambda_inv_for_phase3": use_for_phase3,
        "decision_note": decision_note,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(rollup, fh, indent=2)
    print(json.dumps(rollup, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
