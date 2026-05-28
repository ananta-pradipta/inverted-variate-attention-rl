"""Roll up the SIA regime-wired Phase 1 mini-sweep + smoke into a single JSON.

Reads:
- outputs/sp500/layer2_sia/smoke_regime/summary/fold1_seed42.json
- outputs/sp500/layer2_sia/minisweep_regime_beta_1e-4/summary/fold1_seed{42..46}.json

Emits a single dict to stdout (and an optional --out path) with:
- smoke parity check (ent_coef, log_prob, exposure_mean, aux_inv) at beta_kl=1e-3
- regime-wired mini-sweep per-seed sharpe at beta_kl=1e-4
- regime-wired mean +/- sd
- per-seed delta vs canonical SAC (hardcoded), post-fix-without-I
  (hardcoded), and UR (hardcoded)
- verdict band based on the mean shift relative to post-fix-without-I (+0.329)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

# Canonical SAC fold-1 per-cell pooled Sharpe (from
# invar_rl/results/stage3_rl_ablation/equal_l2/sac/fold1_seed{42..46}.json
# as quoted in the audit report; same reference as the post-fix rollup).
CANONICAL_SAC: Dict[int, float] = {
    42: -0.470, 43: +1.932, 44: +1.554, 45: -0.010, 46: +1.270,
}

# Post-fix SIA WITHOUT the "I" wired (commit 1e2d717, regime_lookup=None).
POSTFIX_NO_I: Dict[int, float] = {
    42: -0.6985, 43: +0.1231, 44: +0.8381, 45: +0.0420, 46: +1.3405,
}

# UR mini-sweep (per audit report; carried for delta-vs-UR column).
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
    """Band the regime-wired mean against post-fix-without-I + canonical.

    Reference: post-fix-without-I mean = +0.329; canonical = +0.855.
    """
    no_i_mean = sum(POSTFIX_NO_I.values()) / len(POSTFIX_NO_I)
    canon = sum(CANONICAL_SAC.values()) / len(CANONICAL_SAC)
    delta_no_i = post_mean - no_i_mean
    delta_canon = post_mean - canon
    if post_mean >= canon - 0.05:
        return (
            f"REGIME I RESCUES: regime-wired mean {post_mean:+.4f} >= "
            f"canonical {canon:+.4f} - 0.05; the I of SIA closes the gap."
        )
    if delta_no_i >= 0.10:
        return (
            f"I HELPS: regime-wired mean {post_mean:+.4f} > "
            f"post-fix-without-I {no_i_mean:+.4f} by >= +0.10; the I "
            f"contributes a meaningful lift but residual gap to "
            f"canonical {canon:+.4f} = {delta_canon:+.4f} remains."
        )
    if abs(delta_no_i) < 0.10:
        return (
            f"I NEUTRAL: regime-wired mean {post_mean:+.4f} is within "
            f"+-0.10 of post-fix-without-I {no_i_mean:+.4f}; the I term "
            f"is a no-op for ranking purposes (consistent with the gate "
            f"already routing through macro). Residual gap to canonical "
            f"{canon:+.4f} = {delta_canon:+.4f}."
        )
    return (
        f"I HURTS: regime-wired mean {post_mean:+.4f} < "
        f"post-fix-without-I {no_i_mean:+.4f} by {-delta_no_i:+.4f}; the "
        f"invariance penalty is over-regularising the latent."
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
        default="outputs/sp500/layer2_sia/smoke_regime/summary/fold1_seed42.json",
    )
    p.add_argument(
        "--sweep-root",
        default="outputs/sp500/layer2_sia/minisweep_regime_beta_1e-4/summary",
    )
    p.add_argument(
        "--out", default="reports/sia/_phase_1_regime_wired_minisweep_rollup.json"
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
        "aux_inv": float(ts.get("aux_inv", float("nan"))),
        "aux_kl": float(ts.get("aux_kl", float("nan"))),
        "aux_gate_l1": float(ts.get("aux_gate_l1", float("nan"))),
    }

    sweep_root = Path(args.sweep_root)
    per_seed: Dict[int, Dict] = {}
    sharpes: List[float] = []
    aux_invs: List[float] = []
    for seed in (42, 43, 44, 45, 46):
        d = _load_summary(sweep_root / f"fold1_seed{seed}.json")
        if d is None:
            per_seed[seed] = {"status": "missing"}
            continue
        sh = float(d.get("test_pooled_sharpe", float("nan")))
        sharpes.append(sh)
        tsi = d.get("sia_train_stats", {})
        cell_inv = float(tsi.get("aux_inv", float("nan")))
        if not (cell_inv != cell_inv):  # guard against nan
            aux_invs.append(cell_inv)
        per_seed[seed] = {
            "status": "ok",
            "regime_wired_sharpe": sh,
            "canonical_sac": CANONICAL_SAC[seed],
            "postfix_no_i": POSTFIX_NO_I[seed],
            "ur": UR_MINISWEEP[seed],
            "delta_vs_canonical": sh - CANONICAL_SAC[seed],
            "delta_vs_postfix_no_i": sh - POSTFIX_NO_I[seed],
            "delta_vs_ur": sh - UR_MINISWEEP[seed],
            "aux_inv": cell_inv,
            "aux_kl": float(tsi.get("aux_kl", float("nan"))),
            "aux_gate_l1": float(tsi.get("aux_gate_l1", float("nan"))),
            "gate_open_fraction": float(tsi.get("gate_open_fraction", float("nan"))),
            "mu_std": float(tsi.get("mu_std", float("nan"))),
            "exposure_mean": float(tsi.get("exposure_mean", float("nan"))),
            "exposure_std": float(tsi.get("exposure_std", float("nan"))),
            "gate_disp": float(tsi.get("gate_0", float("nan"))),
            "gate_pvol": float(tsi.get("gate_1", float("nan"))),
            "gate_effN": float(tsi.get("gate_2", float("nan"))),
            "gate_macro": float(tsi.get("gate_3", float("nan"))),
            "gate_l1u": float(tsi.get("gate_4", float("nan"))),
        }

    post_mean, post_sd = _mean_sd(sharpes)
    canon_mean = sum(CANONICAL_SAC.values()) / len(CANONICAL_SAC)
    no_i_mean = sum(POSTFIX_NO_I.values()) / len(POSTFIX_NO_I)
    ur_mean = sum(UR_MINISWEEP.values()) / len(UR_MINISWEEP)
    inv_mean = sum(aux_invs) / len(aux_invs) if aux_invs else float("nan")

    rollup = {
        "smoke_parity": smoke_parity,
        "per_seed": per_seed,
        "regime_wired_mean": post_mean,
        "regime_wired_sd": post_sd,
        "n_complete": len(sharpes),
        "canonical_sac_mean": canon_mean,
        "postfix_no_i_mean": no_i_mean,
        "ur_mean": ur_mean,
        "delta_vs_canonical": post_mean - canon_mean,
        "delta_vs_postfix_no_i": post_mean - no_i_mean,
        "delta_vs_ur": post_mean - ur_mean,
        "aux_inv_mean_across_cells": inv_mean,
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
