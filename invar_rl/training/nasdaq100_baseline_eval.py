"""NASDAQ-100 Phase 5.5 baseline eval harness (top-25 L/S + native long-only top-K).

Companion to ``invar_rl/training/baselines_long_only_eval.py`` (which serves
the S&P 500 ``lattice_native`` Phase B audit) for the NASDAQ-100 universe.

For each ``(baseline, fold, seed)`` cell:
  1. Loads the baseline's saved per-(day, ticker) score npz from
     ``--npz-root/{baseline}/fold{F}_seed{S}_predictions.npz`` (the same
     npz schema produced by ``src/baselines/v2_runner.py::save_result``).
  2. Builds the NASDAQ-100 ``lattice_bridge`` for the (fold) so we have a
     canonical tradable mask + 1-day log-return panel that line up with
     the npz arrays.
  3. Computes BOTH protocols against the test-segment days:
       (a) ``sharpe_ls``        : top-25 long / bottom-25 short wrapper
                                   (symmetric to the S&P 500 Panel A wrapper).
       (b) ``sharpe_lo_native`` : authors' native long-only top-K
                                   (FactorVAE 50, MASTER 30,
                                    StockMixer/DySTAGE/SWA-InVAR 25).
  4. Writes ``outputs/nasdaq100/baselines/{baseline}/fold{F}_seed{S}.json``
     with both Sharpe numbers, full per-day stats, and the (panel_kind,
     two_regime_val, panel_end, top_k_ls, top_k_native) config.

Usage (per cell)::

    python -m invar_rl.training.nasdaq100_baseline_eval \\
        --baseline master --fold 1 --seed 42

Usage (sweep all 5 seeds within one fold via single bridge build)::

    python -m invar_rl.training.nasdaq100_baseline_eval \\
        --baseline master --fold 1 --sweep-fold

Usage (rollup only)::

    python -m invar_rl.training.nasdaq100_baseline_eval --rollup
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from src.invar import InVARConfig

from invar_rl.data.lattice_bridge import build_lattice_bridge


# ------------------------------------------------------------------
# Native long-only top-K per the published baseline papers.
# FactorVAE  (Duan  et al., AAAI 2022): long-only top-50 on CSI300 -> 50.
# MASTER     (Li    et al., AAAI 2024): long-only top-30 on CSI300/CSI800 -> 30.
# StockMixer (Fan + Shen, AAAI 2024):   long-only top-K on NASDAQ/NYSE -> 25.
# DySTAGE    (Gu    et al., ICAIF 2024): long-only daily on S&P 500 -> 25.
# SWA-InVAR  (this work; not in literature): -> 25 (matches the
#                                                  long-leg of the
#                                                  top-25 L/S wrapper).
# ------------------------------------------------------------------
_NATIVE_K: Dict[str, int] = {
    "factorvae": 50,
    "master": 30,
    "stockmixer": 25,
    "dystage": 25,
    "swa_invar": 25,
    # InVAR Layer 1 + top-25 L/S wrapper (no QP, no SAC). Produced by
    # invar_rl/training/nasdaq100_invar_l1_wrapper_eval.py, NOT by the
    # npz-based per-cell harness below. Listed here so the rollup picks
    # up its JSONs alongside the other Layer-1 baselines.
    "invar_l1": 25,
}

# Top-25 long / bottom-25 short wrapper (matches the S&P 500 Panel A).
_TOP_K_LS: int = 25

# Baselines runnable via this script's per-cell npz harness (skip invar_l1;
# its scores come from the InVAR Layer-1 ckpt, not from a saved npz).
_BASELINES: tuple[str, ...] = tuple(
    b for b in _NATIVE_K.keys() if b != "invar_l1"
)


def _topk_long_short_portfolio(
    y_hat: np.ndarray,
    tradable: np.ndarray,
    log_returns: np.ndarray,
    day_indices: Iterable[int],
    k: int,
) -> dict:
    """Equal-weight top-k long / bottom-k short, daily rebalance.

    Gross = 2.0, net = 0.0. Schema matches
    ``invar_rl.training.native_ranker_eval._topk_long_short_portfolio``.
    """
    daily: list[float] = []
    gross_hist: list[float] = []
    net_hist: list[float] = []
    per_name = 1.0 / float(k)
    for d in day_indices:
        if d + 1 >= log_returns.shape[0]:
            break
        active = np.nonzero(tradable[d])[0]
        if active.size < 2 * k:
            continue
        scores = y_hat[d, active].astype(np.float64)
        valid = np.isfinite(scores)
        if valid.sum() < 2 * k:
            continue
        scores = np.where(valid, scores, -np.inf)
        order = np.argsort(scores)
        short_local = order[:k]
        long_local = order[-k:]
        w = np.zeros(active.size, dtype=np.float64)
        w[long_local] = per_name
        w[short_local] = -per_name
        r_next = log_returns[d + 1, active]
        r_next = np.where(np.isfinite(r_next), r_next, 0.0)
        daily.append(float((w * r_next).sum()))
        gross_hist.append(float(np.sum(np.abs(w))))
        net_hist.append(float(np.sum(w)))
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
        "k_short": k,
        "protocol": "long_short_topk_wrapper",
    }


def _topk_long_only_portfolio(
    y_hat: np.ndarray,
    tradable: np.ndarray,
    log_returns: np.ndarray,
    day_indices: Iterable[int],
    k: int,
) -> dict:
    """Equal-weight top-k LONG-ONLY portfolio, daily rebalance."""
    daily: list[float] = []
    gross_hist: list[float] = []
    net_hist: list[float] = []
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
        "protocol": "long_only_topk_native",
    }


def _build_bridge_for_fold(
    fold: int,
    panel_end: str,
    two_regime_val: bool,
):
    """Build the NASDAQ-100 lattice bridge for one fold.

    The bridge is invariant across seeds for a fixed fold, so callers
    can reuse a single bridge across all 5 seeds. Built once per call;
    no on-disk cache.
    """
    cfg = InVARConfig(fold=fold, seed=42)
    cfg.panel_kind = "nasdaq100"
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    return build_lattice_bridge(cfg)


def run_one_cell(
    baseline: str,
    fold: int,
    seed: int,
    npz_root: Path,
    output_dir: Path,
    panel_end: str,
    two_regime_val: bool,
    bridge=None,
) -> dict:
    """Evaluate one (baseline, fold, seed) cell under BOTH protocols.

    Skips silently if the output JSON already exists.
    """
    out_path = output_dir / f"fold{fold}_seed{seed}.json"
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)

    if bridge is None:
        bridge = _build_bridge_for_fold(
            fold=fold, panel_end=panel_end,
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

    k_native = _NATIVE_K[baseline]
    res_ls = _topk_long_short_portfolio(
        y_hat=y_hat, tradable=tradable, log_returns=log_returns,
        day_indices=list(bridge.test_idx), k=_TOP_K_LS,
    )
    res_lo = _topk_long_only_portfolio(
        y_hat=y_hat, tradable=tradable, log_returns=log_returns,
        day_indices=list(bridge.test_idx), k=k_native,
    )

    print(
        f"  [{baseline}] fold={fold} seed={seed} "
        f"L/S(k={_TOP_K_LS}) sharpe={res_ls['sharpe_annualised']:+.3f} "
        f"  L-only(k={k_native}) sharpe={res_lo['sharpe_annualised']:+.3f}"
    )

    payload = {
        "baseline": baseline,
        "universe": "nasdaq100",
        "fold": fold,
        "seed": seed,
        "n_test_days": int(len(bridge.test_idx)),
        "top_k_ls": _TOP_K_LS,
        "top_k_native": k_native,
        "sharpe_ls": res_ls["sharpe_annualised"],
        "sharpe_lo_native": res_lo["sharpe_annualised"],
        "methods": {
            f"long_short_top{_TOP_K_LS}_wrapper": res_ls,
            f"long_only_top{k_native}_native": res_lo,
        },
        "config": {
            "panel_kind": "nasdaq100",
            "two_regime_val": two_regime_val,
            "panel_end": panel_end,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[ndx-baseline-eval] wrote {out_path}")
    return payload


def rollup(
    output_dir_root: Path,
    baselines: Iterable[str] = tuple(_NATIVE_K.keys()),
    folds: Iterable[int] = (1, 2, 3, 4, 5),
    seeds: Iterable[int] = (42, 43, 44, 45, 46),
) -> dict:
    """Aggregate per-cell JSONs into per-fold-mean + pooled Sharpe for
    BOTH protocols (L/S top-25 wrapper, native long-only top-K)."""
    pooled: dict = {}
    for baseline in baselines:
        b_dir = Path(output_dir_root) / baseline
        if not b_dir.exists():
            print(f"[rollup] {baseline}: directory missing, skipping")
            continue
        per_fold_ls: dict[int, list[float]] = {f: [] for f in folds}
        per_fold_lo: dict[int, list[float]] = {f: [] for f in folds}
        all_ls: list[float] = []
        all_lo: list[float] = []
        missing = 0
        for f in folds:
            for s in seeds:
                p = b_dir / f"fold{f}_seed{s}.json"
                if not p.exists():
                    missing += 1
                    continue
                with open(p) as fh:
                    payload = json.load(fh)
                ls = float(payload["sharpe_ls"])
                lo = float(payload["sharpe_lo_native"])
                per_fold_ls[f].append(ls)
                per_fold_lo[f].append(lo)
                all_ls.append(ls)
                all_lo.append(lo)
        if not all_ls:
            print(f"[rollup] {baseline}: no cells found")
            continue
        pooled[baseline] = {
            "top_k_native": _NATIVE_K[baseline],
            "top_k_ls": _TOP_K_LS,
            "n_cells": len(all_ls),
            "n_missing": missing,
            "pooled_sharpe_ls_mean": float(np.mean(all_ls)),
            "pooled_sharpe_ls_std": float(np.std(all_ls, ddof=1))
            if len(all_ls) > 1 else 0.0,
            "pooled_sharpe_lo_native_mean": float(np.mean(all_lo)),
            "pooled_sharpe_lo_native_std": float(np.std(all_lo, ddof=1))
            if len(all_lo) > 1 else 0.0,
            "per_fold_ls_mean": {
                int(f): float(np.mean(per_fold_ls[f]))
                if per_fold_ls[f] else None
                for f in folds
            },
            "per_fold_lo_native_mean": {
                int(f): float(np.mean(per_fold_lo[f]))
                if per_fold_lo[f] else None
                for f in folds
            },
        }
        print(
            f"[rollup] {baseline:10s} k_native={_NATIVE_K[baseline]:>2d} "
            f"n={pooled[baseline]['n_cells']:>2d} miss={missing}  "
            f"L/S={pooled[baseline]['pooled_sharpe_ls_mean']:+.3f} "
            f"+/- {pooled[baseline]['pooled_sharpe_ls_std']:.3f}  "
            f"L-only={pooled[baseline]['pooled_sharpe_lo_native_mean']:+.3f} "
            f"+/- {pooled[baseline]['pooled_sharpe_lo_native_std']:.3f}"
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
            "NASDAQ-100 Phase 5.5 baseline eval (top-25 L/S wrapper + "
            "native long-only top-K)."
        )
    )
    p.add_argument(
        "--baseline", type=str, choices=list(_BASELINES),
        help="Baseline name. Required for per-cell eval; ignored with --rollup.",
    )
    p.add_argument(
        "--fold", type=int, choices=[1, 2, 3, 4, 5],
        help="Fold (1..5). Required for per-cell eval.",
    )
    p.add_argument(
        "--seed", type=int,
        help="Random seed (42..46). Required unless --sweep-fold is set.",
    )
    p.add_argument(
        "--npz-root", type=str,
        default="results/baselines_nasdaq100_two_regime_val",
        help="Root directory of saved per-(fold, seed) prediction npz.",
    )
    p.add_argument(
        "--output-dir-root", type=str,
        default="outputs/nasdaq100/baselines",
        help="Where to write per-(fold, seed) JSONs and pooled.json.",
    )
    p.add_argument("--panel_end", type=str, default="2025-12-31")
    p.add_argument(
        "--two_regime_val", action="store_true", default=True,
        help="Use the canonical fixed val (2017 H2 + 2018 H2).",
    )
    p.add_argument(
        "--rollup", action="store_true",
        help="Skip per-cell eval and aggregate existing JSONs.",
    )
    p.add_argument(
        "--sweep-fold", action="store_true",
        help=(
            "Run all 5 seeds (42..46) for the given --baseline and "
            "--fold within a single bridge build (5x speedup)."
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
            "Per-cell mode requires --baseline and --fold; add --seed "
            "for one seed or --sweep-fold for all 5 seeds."
        )
    out_dir = output_dir_root / args.baseline

    if args.sweep_fold:
        print(
            f"[ndx-baseline-eval] sweep-fold baseline={args.baseline} "
            f"fold={args.fold} seeds=42..46 (single bridge build)"
        )
        bridge = _build_bridge_for_fold(
            fold=args.fold, panel_end=args.panel_end,
            two_regime_val=args.two_regime_val,
        )
        for seed in (42, 43, 44, 45, 46):
            try:
                run_one_cell(
                    baseline=args.baseline, fold=args.fold, seed=seed,
                    npz_root=Path(args.npz_root), output_dir=out_dir,
                    panel_end=args.panel_end,
                    two_regime_val=args.two_regime_val,
                    bridge=bridge,
                )
            except FileNotFoundError as exc:
                print(f"[ndx-baseline-eval] WARN seed={seed}: {exc}")
        return

    if args.seed is None:
        raise SystemExit(
            "Per-cell mode requires --seed (or --sweep-fold for all 5 seeds)."
        )
    print(
        f"[ndx-baseline-eval] baseline={args.baseline} fold={args.fold} "
        f"seed={args.seed}"
    )
    run_one_cell(
        baseline=args.baseline, fold=args.fold, seed=args.seed,
        npz_root=Path(args.npz_root), output_dir=out_dir,
        panel_end=args.panel_end,
        two_regime_val=args.two_regime_val,
    )


if __name__ == "__main__":
    main()
