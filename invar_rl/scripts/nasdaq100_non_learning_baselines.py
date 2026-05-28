"""Phase 5.5 NASDAQ-100 non-learning baselines (CPU-only).

Computes 5 non-learning strategies analytically from
``data/nasdaq100/prices.parquet`` + ``data/nasdaq100/active_mask.parquet``,
under the same 5-fold macro-stratified protocol used everywhere else in
InVAR-RL on NDX, and writes per-strategy JSON summaries to
``outputs/nasdaq100/baselines/non_learning/{strategy}.json``.

Strategies (matching :mod:`invar_rl.baselines.non_learning`):
  1. ``buy_and_hold``           : equal-weight buy at fold's test-start,
                                  hold to test-end. Re-set per fold.
  2. ``equal_weight_long``      : equal-weight long, daily rebalance.
  3. ``momentum_jt_12_2``       : Jegadeesh-Titman 12-1 momentum (lookback
                                  252 trading days, skip 21), long top
                                  decile / short bottom decile,
                                  monthly rebalance.
  4. ``reversal_1m``            : 1-month reversal (lookback 21), long
                                  bottom decile / short top decile,
                                  weekly rebalance.
  5. ``vol_targeted_market_10`` : equal-weight long market scaled to
                                  10% annualised vol.

For each strategy this script aggregates daily log returns across all
5 folds' test segments, pools them into one strip, then computes the
canonical annualised Sharpe (252-day convention) + final equity + per-
fold means. Output schema mirrors the existing ranker baselines so the
rollup script can read all strategies uniformly.

CPU-only; runs locally in <60 seconds on the full NDX panel.

Usage::

    python invar_rl/scripts/nasdaq100_non_learning_baselines.py

Output: ``outputs/nasdaq100/baselines/non_learning/{strategy}.json``
        ``outputs/nasdaq100/baselines/non_learning/pooled.json``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.invar import InVARConfig

from invar_rl.baselines.non_learning import (
    buy_and_hold,
    equal_weight_long,
    momentum_long_short,
    reversal_long_short,
    volatility_targeted_market,
)
from invar_rl.data.lattice_bridge import build_lattice_bridge


_STRATEGIES = (
    "buy_and_hold",
    "equal_weight_long",
    "momentum_jt_12_2",
    "reversal_1m",
    "vol_targeted_market_10",
)


def _run_one_fold(fold: int) -> dict:
    """Build the NDX bridge for the fold and run all 5 strategies.

    Returns a dict ``{strategy: BaselineResult}`` (the raw dataclass, not
    as_dict, so we can later pool the per-fold daily_log_returns strips).
    """
    cfg = InVARConfig(fold=fold, seed=42)
    cfg.panel_kind = "nasdaq100"
    cfg.two_regime_val = True
    cfg.panel_end = "2025-12-31"
    bridge = build_lattice_bridge(cfg)
    day_indices = list(bridge.test_idx)
    print(
        f"[ndx-non-learning] fold={fold} n_test_days={len(day_indices)} "
        f"N_tickers={len(bridge.tickers)}"
    )

    return {
        "buy_and_hold": buy_and_hold(bridge, day_indices),
        "equal_weight_long": equal_weight_long(bridge, day_indices),
        "momentum_jt_12_2": momentum_long_short(
            bridge, day_indices, lookback=252, skip=21,
        ),
        "reversal_1m": reversal_long_short(
            bridge, day_indices, lookback=21,
        ),
        "vol_targeted_market_10": volatility_targeted_market(
            bridge, day_indices, target_ann_vol=0.10,
        ),
    }


def _pool_sharpe(daily_returns_per_fold: list[np.ndarray]) -> dict:
    """Pool daily log returns across all 5 folds and compute annualised
    Sharpe + final equity. Pools by concatenation; final equity is the
    cumulative product over the concatenated strip.
    """
    if not daily_returns_per_fold:
        return {
            "pooled_sharpe": 0.0, "pooled_mean": 0.0,
            "pooled_vol": 0.0, "pooled_final_equity": 1.0,
            "n_pooled_days": 0,
        }
    arr = np.concatenate(daily_returns_per_fold)
    arr = arr[np.isfinite(arr)]
    mean = float(arr.mean()) if arr.size else 0.0
    vol = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ann_ret = mean * 252.0
    ann_vol = vol * np.sqrt(252.0)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    return {
        "pooled_sharpe": sharpe,
        "pooled_mean": mean,
        "pooled_vol": vol,
        "pooled_final_equity": float(np.exp(arr.sum())) if arr.size else 1.0,
        "n_pooled_days": int(arr.size),
    }


def main() -> int:
    out_root = Path("outputs/nasdaq100/baselines/non_learning")
    out_root.mkdir(parents=True, exist_ok=True)

    # Run all 5 folds once; collect per-fold results per strategy.
    per_fold_results: dict[int, dict] = {}
    for fold in (1, 2, 3, 4, 5):
        per_fold_results[fold] = _run_one_fold(fold)

    pooled = {}
    for strat in _STRATEGIES:
        per_fold_sharpe = {
            fold: float(per_fold_results[fold][strat].sharpe_annualised)
            for fold in per_fold_results
        }
        per_fold_mean = {
            fold: float(per_fold_results[fold][strat].mean_return)
            for fold in per_fold_results
        }
        per_fold_eq = {
            fold: float(per_fold_results[fold][strat].final_equity)
            for fold in per_fold_results
        }
        daily_per_fold = [
            np.asarray(
                per_fold_results[fold][strat].daily_log_returns,
                dtype=np.float64,
            )
            for fold in per_fold_results
        ]
        pool_block = _pool_sharpe(daily_per_fold)
        pool_block["pool_method"] = "concat_daily"

        payload = {
            "strategy": strat,
            "universe": "nasdaq100",
            "protocol": (
                "deterministic; 5-fold macro-stratified test segments; "
                "two_regime_val=True; panel_end=2025-12-31; CPU-only "
                "from data/nasdaq100/prices.parquet."
            ),
            "per_fold_sharpe": per_fold_sharpe,
            "per_fold_mean_return": per_fold_mean,
            "per_fold_final_equity": per_fold_eq,
            "pooled": pool_block,
            "fold_details": {
                str(f): per_fold_results[f][strat].as_dict()
                for f in per_fold_results
            },
        }
        out_path = out_root / f"{strat}.json"
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(
            f"[ndx-non-learning] {strat:24s}  pooled Sharpe="
            f"{pool_block['pooled_sharpe']:+.3f}  "
            f"per-fold mean Sharpe="
            f"{float(np.mean(list(per_fold_sharpe.values()))):+.3f}"
        )
        pooled[strat] = {
            "per_fold_sharpe": per_fold_sharpe,
            "pooled_sharpe": pool_block["pooled_sharpe"],
            "pool_method": pool_block["pool_method"],
        }

    pooled_path = out_root / "pooled.json"
    with open(pooled_path, "w") as fh:
        json.dump(pooled, fh, indent=2, default=str)
    print(f"[ndx-non-learning] wrote {pooled_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
