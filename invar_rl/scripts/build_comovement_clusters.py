"""Pre-build A2 co-movement cluster caches for a given universe + folds.

For each requested fold:

1. Build the canonical panel via ``src.baselines.v2_runner.build_panel``
   for the requested ``--panel-kind`` (e.g. ``nasdaq100``,
   ``biotech_nbi_enriched``, ``lattice_native``).
2. Apply the canonical tradable mask and fold split (two_regime_val).
3. Construct the TRAIN-segment daily-returns matrix from feature 0
   (log returns) with NaN-masked zero-fill rows, exactly as the JIT
   path in ``run_stage1_pretrain`` does.
4. Fit a ``CoMovementClusterer`` with the configured K / window / seed.
5. Persist via ``save_cluster_ids`` to
   ``cache/pretrain_improvements/comovement/<universe>/foldF/cluster_ids.parquet``.

This is a thin local LOCAL VERIFY helper so the per-fold clusters can be
audited (non-degeneracy: every cluster size >= 5) before any Wulver
submit. The A2 trainer (READ-ONLY here) will reuse the cached parquet
on hit and skip the fit.

Usage::

    PYTHONPATH=$PWD python -m invar_rl.scripts.build_comovement_clusters \\
        --panel-kind nasdaq100 \\
        --universe nasdaq100 \\
        --folds 1 2 3 4 5

    PYTHONPATH=$PWD python -m invar_rl.scripts.build_comovement_clusters \\
        --panel-kind biotech_nbi_enriched \\
        --universe biotech_nbi_enriched \\
        --folds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MIN_CLUSTER_FLOOR = 5


def _build_returns_for_fold(
    panel_kind: str,
    fold: int,
    panel_end: str,
) -> tuple[pd.DataFrame, list[str], int]:
    """Build the TRAIN-segment daily-returns DataFrame for one fold.

    Returns:
        train_df: DataFrame indexed by date with one column per ticker.
        tickers: list of ticker symbols (same order as panel columns).
        n_train_days: number of TRAIN rows.
    """
    from src.baselines.train_invar_stx_v2 import InvarSTXV2Config
    from src.baselines.v2_runner import (
        build_masks,
        build_panel,
        fold_split,
    )

    cfg = InvarSTXV2Config(fold=int(fold), seed=42)
    cfg.panel_kind = str(panel_kind)
    cfg.two_regime_val = True
    cfg.panel_end = str(panel_end)
    cfg.enable_retrieval_bank = False

    x_raw, _, tickers, dates = build_panel(cfg)
    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    train_idx, _val_idx, _test_idx = fold_split(cfg, dates)

    train_rets = x_raw[train_idx, :, 0].astype(np.float64)
    train_mask_pan = tradable[train_idx]
    train_rets = np.where(train_mask_pan, train_rets, np.nan)
    train_dates_pd = pd.to_datetime(np.asarray(dates)[train_idx])
    train_df = pd.DataFrame(
        train_rets,
        index=train_dates_pd,
        columns=list(tickers),
    )
    return train_df, list(tickers), int(train_df.shape[0])


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Pre-build A2 co-movement cluster caches for a universe."
        )
    )
    p.add_argument(
        "--panel-kind", type=str, required=True,
        choices=[
            "biotech", "lattice_native", "nasdaq100",
            "djia30", "biotech_nbi", "biotech_nbi_enriched",
        ],
        help="Panel kind to build (passed to build_panel).",
    )
    p.add_argument(
        "--universe", type=str, required=True,
        help=(
            "Cache key under cache/pretrain_improvements/comovement/"
            "<universe>/foldF/."
        ),
    )
    p.add_argument(
        "--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
    )
    p.add_argument("--panel-end", type=str, default="2025-12-31")
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--window", type=int, default=252)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--force", action="store_true",
        help="Refit even if the per-fold parquet already exists.",
    )
    args = p.parse_args()

    from src.models.pretrain_improvements.comovement_clustering import (
        CoMovementClusterer,
        CoMovementConfig,
        cluster_ids_path,
        cluster_size_summary,
        save_cluster_ids,
    )

    print(
        f"[A2 build] panel_kind={args.panel_kind} "
        f"universe={args.universe} K={args.n_clusters} "
        f"window={args.window} seed={args.seed} folds={args.folds}",
        flush=True,
    )
    any_degenerate = False
    for f in args.folds:
        out_path = cluster_ids_path(universe=args.universe, fold=int(f))
        if out_path.exists() and not args.force:
            cached = pd.read_parquet(out_path)
            sizes = cluster_size_summary(
                dict(
                    zip(
                        cached["ticker"].astype(str),
                        cached["cluster_id"].astype(int),
                    )
                )
            )
            min_sz = min(sizes.values())
            tag = "[OK]" if min_sz >= MIN_CLUSTER_FLOOR else "[ERR]"
            print(
                f"[A2 build] {tag} F{f} cached at {out_path} "
                f"N={len(cached)} K={cached['cluster_id'].nunique()} "
                f"sizes={list(sizes.values())} min={min_sz}",
                flush=True,
            )
            if min_sz < MIN_CLUSTER_FLOOR:
                any_degenerate = True
            continue

        train_df, tickers, n_train = _build_returns_for_fold(
            panel_kind=args.panel_kind, fold=int(f),
            panel_end=args.panel_end,
        )
        print(
            f"[A2 build] F{f} TRAIN: T={n_train} N={len(tickers)} "
            f"(panel_kind={args.panel_kind})",
            flush=True,
        )
        clusterer = CoMovementClusterer(
            CoMovementConfig(
                universe=args.universe,
                n_clusters=int(args.n_clusters),
                window=int(args.window),
                seed=int(args.seed),
            )
        )
        cluster_ids = clusterer.fit(
            train_df, n_clusters=int(args.n_clusters),
        )
        sizes = cluster_size_summary(cluster_ids)
        min_sz = min(sizes.values())
        save_cluster_ids(
            cluster_ids=cluster_ids,
            universe=args.universe,
            fold=int(f),
            n_clusters=int(args.n_clusters),
            n_train_days=n_train,
            n_windows=int(clusterer.n_windows_),
            seed=int(args.seed),
        )
        tag = "[OK]" if min_sz >= MIN_CLUSTER_FLOOR else "[ERR]"
        print(
            f"[A2 build] {tag} F{f} fit N={len(cluster_ids)} "
            f"K={int(args.n_clusters)} sizes={list(sizes.values())} "
            f"min={min_sz} -> {out_path}",
            flush=True,
        )
        if min_sz < MIN_CLUSTER_FLOOR:
            any_degenerate = True

    if any_degenerate:
        print(
            "[A2 build] [ERR] at least one fold has a cluster with "
            f"size < {MIN_CLUSTER_FLOOR}; re-tune K or window before "
            "submitting.",
            flush=True,
        )
        return 1
    print(
        f"[A2 build] [OK] all folds have min cluster size >= "
        f"{MIN_CLUSTER_FLOOR}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
