"""Native long-only top-K Sharpe for Layer-1 ranker baselines.

This is the COMPANION to ``invar_rl/training/native_ranker_eval.py``
(top-K L/S wrapper) and audits Phase B of the InVAR-RL baseline review:
re-evaluating four ranker baselines under their AUTHORS' native long-only
top-K protocols on our S&P 500 ``lattice_native`` panel, so the reader
can confirm the L/S wrapper used in Panel A is not biasing the headline
comparison.

Native protocols (per the published papers, transplanted to our panel):
  FactorVAE  (Duan  et al., AAAI 2022): long-only top-50 on CSI300 -> top-50 here.
  MASTER     (Li    et al., AAAI 2024): long-only top-30 on CSI300/CSI800 -> top-30 here.
  StockMixer (Fan + Shen, AAAI 2024):  long-only top-K on NASDAQ/NYSE/S&P 500 -> top-25 here.
  DySTAGE    (Gu    et al., ICAIF 2024): long-only daily on S&P 500 -> top-25 here.

For each (fold, seed) cell, the script:
  1. Loads the baseline's saved predictions from
     ``--npz-root/{baseline}/fold{F}_seed{S}_predictions.npz`` (the same
     npz produced by ``src/baselines/v2_runner.py::save_result`` and
     consumed by ``invar_rl.training.native_ranker_eval``).
  2. For each test day, picks the top-K tradable stocks by predicted
     score, weights them equally (gross = 1.0, net = +1.0), and applies
     the next-day realised 1-day log return.
  3. Computes annualised Sharpe with the same 252-day convention used
     by the L/S wrapper and the RL stack.

Output: ``--output-dir-root/{baseline}/fold{F}_seed{S}.json`` (one
JSON per cell), and a top-level ``pooled.json`` that aggregates across
all 25 cells per baseline once every cell is written.

Usage (single cell):

    python -m invar_rl.training.baselines_long_only_eval \
        --baseline master --fold 1 --seed 42 --top-k 30

Usage (rollup only, after all 25 cells per baseline are written):

    python -m invar_rl.training.baselines_long_only_eval --rollup

The matching Wulver sbatch is
``invar_rl/scripts/wulver/baselines_long_only_eval.sbatch``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from src.invar import InVARConfig

from invar_rl.data.lattice_bridge import build_lattice_bridge


# Native long-only top-K per the published paper, transplanted to our panel.
_BASELINE_NATIVE_K: dict[str, int] = {
    "factorvae": 50,
    "master": 30,
    "stockmixer": 25,
    "dystage": 25,
}

_BASELINES: tuple[str, ...] = tuple(_BASELINE_NATIVE_K.keys())


def _topk_long_only_portfolio(
    y_hat: np.ndarray,
    tradable: np.ndarray,
    log_returns: np.ndarray,
    day_indices: Iterable[int],
    k: int,
) -> dict:
    """Equal-weight top-k LONG-ONLY portfolio, daily rebalance.

    Args:
        y_hat: ``(T, N)`` per-day predicted scores.
        tradable: ``(T, N)`` bool mask of tradeable stocks per day.
        log_returns: ``(T, N)`` realised 1-day log return per stock.
        day_indices: trading-day indices on which to trade (test segment).
        k: number of long positions (long-only; no short leg).

    Returns:
        Dict with mean / vol / annualised Sharpe / final equity / exposure
        stats; identical schema (minus the L/S-specific fields) to the
        L/S wrapper in ``invar_rl.training.native_ranker_eval``.
    """
    daily: list[float] = []
    gross_hist: list[float] = []
    net_hist: list[float] = []
    n_long: list[int] = []
    per_name = 1.0 / float(k)
    for d in day_indices:
        if d + 1 >= log_returns.shape[0]:
            break
        active = np.nonzero(tradable[d])[0]
        if active.size < k:
            continue
        scores = y_hat[d, active].astype(np.float64)
        valid = np.isfinite(scores)
        if valid.sum() < k:
            continue
        # Mask invalid scores so they never make it into the top-k.
        scores = np.where(valid, scores, -np.inf)
        order = np.argsort(scores)
        long_local = order[-k:]
        w = np.zeros(active.size, dtype=np.float64)
        w[long_local] = per_name
        r_next = log_returns[d + 1, active]
        r_next = np.where(np.isfinite(r_next), r_next, 0.0)
        daily.append(float((w * r_next).sum()))
        gross_hist.append(float(np.sum(np.abs(w))))
        net_hist.append(float(np.sum(w)))
        n_long.append(k)
    arr = np.asarray(daily)
    arr = arr[np.isfinite(arr)]
    mean = float(arr.mean()) if arr.size else 0.0
    vol = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ann_ret = mean * 252.0
    ann_vol = vol * np.sqrt(252.0)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    return {
        "mean_return": mean,
        "volatility": vol,
        "sharpe_annualised": sharpe,
        "final_equity": float(np.exp(arr.sum())) if arr.size else 1.0,
        "n_steps": int(arr.size),
        "gross_exposure_mean": (
            float(np.mean(gross_hist)) if gross_hist else 0.0
        ),
        "net_exposure_mean": (
            float(np.mean(net_hist)) if net_hist else 0.0
        ),
        "k_long": k,
        "k_short": 0,
        "protocol": "long_only_topk",
        # Per-day log-return series for the daily cumulative-return figure
        # (Figure 8); summary stats above are recomputable from this.
        "daily_log_returns": [float(x) for x in arr.tolist()],
    }


def _build_bridge_for_fold(
    fold: int,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
):
    """Build the lattice bridge for one fold.

    The bridge depends on (panel_kind, panel_end, fold, two_regime_val)
    only; the seed in ``InVARConfig`` is unused by ``build_panel`` /
    ``build_masks`` / ``fold_split``. So we can reuse the same bridge
    object across all 5 seeds of one fold, saving ~80% of the
    per-cell wall time which is dominated by the bridge build.
    """
    cfg = InVARConfig(fold=fold, seed=42)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    return build_lattice_bridge(cfg)


def run_one_cell(
    baseline: str,
    fold: int,
    seed: int,
    npz_root: Path,
    output_dir: Path,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
    top_k: int,
    bridge=None,
) -> dict:
    """Evaluate one (baseline, fold, seed) cell under long-only top-K.

    Skips silently if the output JSON already exists (idempotent rerun).
    Pass ``bridge`` to reuse an already-built lattice bridge across
    seeds of the same fold.
    """
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)

    if bridge is None:
        bridge = _build_bridge_for_fold(
            fold=fold, panel_kind=panel_kind, panel_end=panel_end,
            two_regime_val=two_regime_val,
        )

    npz_path = (
        Path(npz_root) / baseline
        / f"fold{fold}_seed{seed}_predictions.npz"
    )
    if not npz_path.exists():
        raise FileNotFoundError(
            f"baseline predictions not found: {npz_path}"
        )
    blob = np.load(npz_path, allow_pickle=False)
    y_hat = blob["y_hat"]
    tradable = blob["tradable_mask"]
    log_returns = bridge.log_returns_1d

    if y_hat.shape != log_returns.shape:
        raise ValueError(
            f"baseline {baseline} y_hat shape {y_hat.shape} "
            f"!= bridge log_returns shape {log_returns.shape}"
        )

    res = _topk_long_only_portfolio(
        y_hat=y_hat,
        tradable=tradable,
        log_returns=log_returns,
        day_indices=list(bridge.test_idx),
        k=top_k,
    )
    print(
        f"  long_only_top{top_k:3d}  sharpe={res['sharpe_annualised']:+.3f} "
        f"ann_ret={res['mean_return']*252:+.4f} "
        f"ann_vol={res['volatility']*(252**0.5):+.4f} "
        f"eq={res['final_equity']:.4f}"
    )

    payload = {
        "baseline": baseline,
        "fold": fold,
        "seed": seed,
        "model": (
            f"Native ranker baseline L1={baseline} -> "
            f"long-only top-{top_k} equal-weight daily rebalance "
            f"(authors' native eval protocol; no L/S wrapper, no RL)"
        ),
        "n_test_days": int(len(bridge.test_idx)),
        "top_k": top_k,
        "methods": {f"long_only_top{top_k}": res},
        "config": {
            "panel_kind": panel_kind,
            "two_regime_val": two_regime_val,
            "panel_end": panel_end,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[long-only eval] wrote {out_path}")
    return payload


def rollup(
    output_dir_root: Path,
    baselines: Iterable[str] = _BASELINES,
    folds: Iterable[int] = (1, 2, 3, 4, 5),
    seeds: Iterable[int] = (42, 43, 44, 45, 46),
) -> dict:
    """Aggregate per-cell JSONs into per-fold-mean + pooled Sharpe.

    Pools across all (fold, seed) cells equally, matching the
    convention of ``invar_rl/training/native_ranker_eval`` and the
    Panel A wrapper Sharpe in the InVAR-RL paper.
    """
    pooled: dict = {}
    for baseline in baselines:
        b_dir = Path(output_dir_root) / baseline
        if not b_dir.exists():
            print(f"[rollup] {baseline}: directory missing, skipping")
            continue
        per_fold: dict[int, list[float]] = {f: [] for f in folds}
        all_cells: list[float] = []
        missing = 0
        for f in folds:
            for s in seeds:
                p = b_dir / f"fold{f}_seed{s}.json"
                if not p.exists():
                    missing += 1
                    continue
                with open(p) as fh:
                    payload = json.load(fh)
                method_key = next(iter(payload["methods"]))
                sharpe = float(
                    payload["methods"][method_key]["sharpe_annualised"]
                )
                per_fold[f].append(sharpe)
                all_cells.append(sharpe)
        if not all_cells:
            print(f"[rollup] {baseline}: no cells found")
            continue
        pooled[baseline] = {
            "top_k": _BASELINE_NATIVE_K.get(baseline),
            "n_cells": len(all_cells),
            "n_missing": missing,
            "pooled_sharpe_mean": float(np.mean(all_cells)),
            "pooled_sharpe_std": float(np.std(all_cells, ddof=1))
            if len(all_cells) > 1 else 0.0,
            "per_fold_mean": {
                int(f): float(np.mean(per_fold[f])) if per_fold[f] else None
                for f in folds
            },
        }
        print(
            f"[rollup] {baseline:10s} k={pooled[baseline]['top_k']:>3d} "
            f"n={pooled[baseline]['n_cells']:>2d} miss={missing} "
            f"pooled Sharpe={pooled[baseline]['pooled_sharpe_mean']:+.3f} "
            f"+/- {pooled[baseline]['pooled_sharpe_std']:.3f}"
        )
    out_path = Path(output_dir_root) / "pooled.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(pooled, fh, indent=2)
    print(f"[rollup] wrote {out_path}")
    return pooled


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Native long-only top-K Sharpe eval for Layer-1 ranker "
            "baselines (Phase B of the InVAR-RL baseline audit)."
        )
    )
    p.add_argument(
        "--baseline", type=str, choices=list(_BASELINES),
        help=(
            "Baseline name. Required for per-cell eval; ignored "
            "when --rollup is set."
        ),
    )
    p.add_argument(
        "--fold", type=int, choices=[1, 2, 3, 4, 5],
        help="Fold index (1..5). Required for per-cell eval.",
    )
    p.add_argument(
        "--seed", type=int,
        help="Random seed (42..46). Required for per-cell eval.",
    )
    p.add_argument(
        "--top-k", type=int, default=None,
        help=(
            "Top-K for the long-only portfolio. If omitted, uses the "
            "baseline's native K (FactorVAE 50, MASTER 30, "
            "StockMixer 25, DySTAGE 25)."
        ),
    )
    p.add_argument(
        "--npz-root", type=str,
        default="results/baselines_universal_two_regime_val",
        help="Root directory of saved per-(fold, seed) prediction npz.",
    )
    p.add_argument(
        "--output-dir-root", type=str,
        default="invar_rl/results/baselines_long_only",
        help="Where to write per-(fold, seed) JSONs and pooled.json.",
    )
    p.add_argument(
        "--panel_kind", type=str, default="lattice_native",
        choices=["biotech", "lattice_native"],
    )
    p.add_argument("--panel_end", type=str, default="2025-12-31")
    p.add_argument("--two_regime_val", action="store_true", default=True)
    p.add_argument(
        "--rollup", action="store_true",
        help=(
            "Skip per-cell eval and just aggregate existing JSONs into "
            "pooled.json."
        ),
    )
    p.add_argument(
        "--sweep-fold", action="store_true",
        help=(
            "Run all 5 seeds (42-46) for the given --baseline and "
            "--fold within a single bridge build. Faster than 5 "
            "independent invocations because the lattice bridge is "
            "the slow piece (~20s) and is invariant across seeds for "
            "a fixed fold."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir_root = Path(args.output_dir_root)
    if args.rollup:
        rollup(output_dir_root=output_dir_root)
        return
    if args.baseline is None or args.fold is None:
        raise SystemExit(
            "Per-cell mode requires --baseline and --fold; "
            "add --seed for one seed, --sweep-fold for all 5 seeds, "
            "or use --rollup for aggregation."
        )
    top_k = (
        int(args.top_k) if args.top_k is not None
        else _BASELINE_NATIVE_K[args.baseline]
    )
    out_dir = output_dir_root / args.baseline

    if args.sweep_fold:
        # Build the bridge once for this fold; iterate over the 5 seeds.
        # The bridge is invariant across seeds; this is a ~5x speedup
        # over 5 independent invocations.
        print(
            f"[long-only eval] sweep-fold baseline={args.baseline} "
            f"fold={args.fold} seeds=42..46 top_k={top_k} "
            f"(single bridge build)"
        )
        bridge = _build_bridge_for_fold(
            fold=args.fold, panel_kind=args.panel_kind,
            panel_end=args.panel_end, two_regime_val=args.two_regime_val,
        )
        for seed in (42, 43, 44, 45, 46):
            print(f"[long-only eval] -- fold={args.fold} seed={seed}")
            run_one_cell(
                baseline=args.baseline,
                fold=args.fold,
                seed=seed,
                npz_root=Path(args.npz_root),
                output_dir=out_dir,
                panel_kind=args.panel_kind,
                panel_end=args.panel_end,
                two_regime_val=args.two_regime_val,
                top_k=top_k,
                bridge=bridge,
            )
        return

    if args.seed is None:
        raise SystemExit(
            "Per-cell mode requires --seed (or --sweep-fold for all "
            "5 seeds of one fold)."
        )
    print(
        f"[long-only eval] baseline={args.baseline} fold={args.fold} "
        f"seed={args.seed} top_k={top_k}"
    )
    run_one_cell(
        baseline=args.baseline,
        fold=args.fold,
        seed=args.seed,
        npz_root=Path(args.npz_root),
        output_dir=out_dir,
        panel_kind=args.panel_kind,
        panel_end=args.panel_end,
        two_regime_val=args.two_regime_val,
        top_k=top_k,
    )


if __name__ == "__main__":
    main()
