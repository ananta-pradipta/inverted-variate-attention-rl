"""Roll up the SIA post-fix Phase 1 mini-sweep + smoke into a single JSON.

Reads:
- outputs/sp500/layer2_sia/smoke_postfix/summary/fold1_seed42.json
- outputs/sp500/layer2_sia/minisweep_postfix_beta_1e-4/summary/fold1_seed{42..46}.json

Emits a single dict to stdout (and an optional --out path) with:
- smoke parity check (ent_coef, log_prob, exposure_mean) at beta_kl=1e-3
- post-fix mini-sweep per-seed sharpe at beta_kl=1e-4
- post-fix mean +/- sd
- per-seed delta vs canonical SAC (hardcoded) and pre-fix SIA (hardcoded)

The canonical SAC and pre-fix SIA per-cell sharpes are baked in from the
audit report; the UR mini-sweep is also baked in for the verdict bands.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

# Canonical SAC fold-1 per-cell pooled Sharpe (from
# invar_rl/results/stage3_rl_ablation/equal_l2/sac/fold1_seed{42..46}.json
# as quoted in the audit report).
CANONICAL_SAC: Dict[int, float] = {
    42: -0.470, 43: +1.932, 44: +1.554, 45: -0.010, 46: +1.270,
}

# Pre-fix SIA mini-sweep at beta_kl=1e-4 (commit 72723ca summary).
PREFIX_SIA: Dict[int, float] = {
    42: -0.541, 43: -0.371, 44: -0.340, 45: -0.510, 46: +1.321,
}

# UR mini-sweep (per caller's task prompt).
UR_MINISWEEP: Dict[int, float] = {
    42: -0.493, 43: +0.117, 44: +0.195, 45: -0.032, 46: +1.294,
}


def _mean_sd(xs: List[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def _verdict(post_mean: float) -> str:
    canon = sum(CANONICAL_SAC.values()) / len(CANONICAL_SAC)
    delta = post_mean - canon
    if abs(delta) <= 0.10:
        return (
            f"BUGS WERE THE ISSUE: post-fix mean {post_mean:+.4f} is within "
            f"+-0.10 of canonical SAC {canon:+.4f}; SIA architecture works."
        )
    if 0.20 <= post_mean <= 0.70:
        return (
            f"IMPROVED, RESIDUAL GAP: post-fix mean {post_mean:+.4f} sits in "
            f"[+0.20, +0.70] but does not match canonical {canon:+.4f}; "
            f"architecture has residual issues."
        )
    if post_mean < 0.20:
        return (
            f"BUGS NECESSARY BUT NOT SUFFICIENT: post-fix mean "
            f"{post_mean:+.4f} < +0.20; the SIA architectural concern "
            f"persists even after B1+B2+B3 fixes."
        )
    return (
        f"POST-FIX > CANONICAL: post-fix mean {post_mean:+.4f} > canonical "
        f"{canon:+.4f}; investigate as a potential genuine improvement "
        f"(verify protocol parity before claiming)."
    )


def _load_summary(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--smoke-summary",
        default="outputs/sp500/layer2_sia/smoke_postfix/summary/fold1_seed42.json",
    )
    p.add_argument(
        "--sweep-root",
        default="outputs/sp500/layer2_sia/minisweep_postfix_beta_1e-4/summary",
    )
    p.add_argument(
        "--out", default="reports/sia/_phase_1_postfix_minisweep_rollup.json"
    )
    args = p.parse_args()

    smoke = _load_summary(Path(args.smoke_summary)) or {}
    ts = smoke.get("sia_train_stats", {})
    smoke_parity = {
        "smoke_summary_path": args.smoke_summary,
        "best_val_sharpe": float(smoke.get("best_val_sharpe", float("nan"))),
        "test_pooled_sharpe": float(smoke.get("test_pooled_sharpe", float("nan"))),
        "ent_coef": float(ts.get("ent_coef", float("nan"))),
        "log_prob_mean": float(ts.get("log_prob", float("nan"))),
        "exposure_mean": float(ts.get("exposure_mean", float("nan"))),
        "exposure_std": float(ts.get("exposure_std", float("nan"))),
        "gate_open_fraction": float(ts.get("gate_open_fraction", float("nan"))),
        "mu_std": float(ts.get("mu_std", float("nan"))),
    }

    sweep_root = Path(args.sweep_root)
    per_seed: Dict[int, Dict] = {}
    sharpes: List[float] = []
    for seed in (42, 43, 44, 45, 46):
        d = _load_summary(sweep_root / f"fold1_seed{seed}.json")
        if d is None:
            per_seed[seed] = {"status": "missing"}
            continue
        sh = float(d.get("test_pooled_sharpe", float("nan")))
        sharpes.append(sh)
        per_seed[seed] = {
            "status": "ok",
            "post_fix_sharpe": sh,
            "canonical_sac": CANONICAL_SAC[seed],
            "pre_fix_sia": PREFIX_SIA[seed],
            "ur": UR_MINISWEEP[seed],
            "delta_vs_canonical": sh - CANONICAL_SAC[seed],
            "delta_vs_prefix": sh - PREFIX_SIA[seed],
            "delta_vs_ur": sh - UR_MINISWEEP[seed],
        }

    post_mean, post_sd = _mean_sd(sharpes)
    canon_mean = sum(CANONICAL_SAC.values()) / len(CANONICAL_SAC)
    prefix_mean = sum(PREFIX_SIA.values()) / len(PREFIX_SIA)
    ur_mean = sum(UR_MINISWEEP.values()) / len(UR_MINISWEEP)

    rollup = {
        "smoke_parity": smoke_parity,
        "per_seed": per_seed,
        "post_fix_mean": post_mean,
        "post_fix_sd": post_sd,
        "post_fix_n_complete": len(sharpes),
        "canonical_sac_mean": canon_mean,
        "pre_fix_sia_mean": prefix_mean,
        "ur_mean": ur_mean,
        "delta_post_vs_canonical": post_mean - canon_mean,
        "delta_post_vs_prefix": post_mean - prefix_mean,
        "delta_post_vs_ur": post_mean - ur_mean,
        "verdict": _verdict(post_mean),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(rollup, fh, indent=2)
    print(json.dumps(rollup, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
