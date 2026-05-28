"""Generic per-universe Layer 1 entrypoint for A3 + A4 Stage-1 pretrain.

A3 (2026-05-27): sequential three-stage curriculum regime -> sector ->
co-movement. A4 (2026-05-27): joint single-stage objective that
aggregates the day-level regime InfoNCE with both per-stock SupCon terms
(sector + co-movement) per batch.

This is a thin (panel_kind, fold, seed) runner. It builds an
``InvarSTXV2Config`` with the A3 / A4 cfg fields set, then calls the
canonical pretrain + finetune from
``src.baselines.train_invar_clpretrain_v2``. The pretrain code itself is
left BYTE-IDENTICAL; this file only sets cfg fields and dispatches.

Per-fold encoder ckpt lives at ``<output-dir>/_ckpt/fold{F}_encoder.pt``
and is shared across the 5 finetune seeds (42..46). The sector cache
lives at ``cache/sector_labels/<sector-universe-id>.parquet`` and the
co-movement cluster cache at
``cache/pretrain_improvements/comovement/<comove-universe-id>/foldF/
cluster_ids.parquet`` (both pre-built; rebuilt JIT on cache miss).

Usage (A3 NASDAQ-100)::

    python -m invar_rl.training.universe_layer1_a3_a4 \\
        --mode a3 --panel-kind nasdaq100 --fold 1 --seed 42 \\
        --output-dir outputs/nasdaq100/layer1_a3_seq_regime_sector_comove \\
        --sector-universe-id nasdaq100 \\
        --comovement-universe-id nasdaq100 \\
        --pretrain-epochs 10 --comovement-epochs 5 --finetune-epochs 10

Usage (A4 biotech NBI)::

    python -m invar_rl.training.universe_layer1_a3_a4 \\
        --mode a4 --panel-kind biotech_nbi_enriched --fold 1 --seed 42 \\
        --output-dir outputs/biotech_nbi_enriched/layer1_a4_joint \\
        --sector-universe-id biotech_nbi_enriched \\
        --comovement-universe-id biotech_nbi_enriched \\
        --pretrain-epochs 10 --finetune-epochs 10

Add ``--pretrain-only`` to run Stage 1 and exit; add ``--skip-pretrain``
to load an existing per-fold encoder ckpt for Stage 2 only.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _build_cfg(args: argparse.Namespace):
    """Build the per-universe InVAR cfg with A3 / A4 flags set."""
    from src.baselines.train_invar_stx_v2 import InvarSTXV2Config

    cfg = InvarSTXV2Config(fold=int(args.fold), seed=int(args.seed))
    cfg.panel_kind = str(args.panel_kind)
    cfg.two_regime_val = True
    cfg.panel_end = str(args.panel_end)
    cfg.output_dir = str(args.output_dir)
    cfg.enable_retrieval_bank = False
    # Shared cohort universe ids (both A3 sector stage and A4 joint read
    # these; A3 comove stage and A4 joint read the comovement id).
    cfg.pretrain_sector_universe_id = str(args.sector_universe_id)
    cfg.pretrain_comovement_universe_id = str(args.comovement_universe_id)
    cfg.pretrain_comovement_n_clusters = int(args.comovement_n_clusters)
    cfg.pretrain_comovement_window = int(args.comovement_window)
    cfg.pretrain_comovement_epochs = int(args.comovement_epochs)
    if args.mode == "a3":
        # Sequential three-stage curriculum.
        cfg.pretrain_stages = ["regime", "sector", "comovement"]
        cfg.pretrain_positive_method = "regime"
    else:
        # A4 joint single-stage objective.
        cfg.pretrain_positive_method = "joint"
        cfg.pretrain_joint_weight_regime = float(args.joint_weight_regime)
        cfg.pretrain_joint_weight_sector = float(args.joint_weight_sector)
        cfg.pretrain_joint_weight_comove = float(args.joint_weight_comove)
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Generic per-universe Layer 1 A3 (sequential regime->sector->"
            "comovement) / A4 (joint) Stage-1 pretrain."
        )
    )
    p.add_argument(
        "--mode", type=str, required=True, choices=["a3", "a4"],
    )
    p.add_argument(
        "--panel-kind", type=str, required=True,
        help="e.g. nasdaq100, biotech_nbi_enriched, lattice_native.",
    )
    p.add_argument("--fold", type=int, required=True,
                   choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--panel-end", type=str, default="2025-12-31")
    p.add_argument("--sector-universe-id", type=str, required=True)
    p.add_argument("--comovement-universe-id", type=str, required=True)
    p.add_argument("--comovement-n-clusters", type=int, default=8)
    p.add_argument("--comovement-window", type=int, default=252)
    p.add_argument("--comovement-epochs", type=int, default=5)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--joint-weight-regime", type=float, default=1.0)
    p.add_argument("--joint-weight-sector", type=float, default=1.0)
    p.add_argument("--joint-weight-comove", type=float, default=1.0)
    p.add_argument("--pretrain-only", action="store_true")
    p.add_argument("--skip-pretrain", action="store_true")
    args = p.parse_args()

    if args.pretrain_only and args.skip_pretrain:
        raise SystemExit(
            "--pretrain-only and --skip-pretrain are mutually exclusive."
        )

    from src.baselines.train_invar_clpretrain_v2 import (
        run_stage1_pretrain,
        run_stage1_sequential_pretrain,
        run_stage2_finetune,
    )

    cfg = _build_cfg(args)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(
        f"[universe_layer1_a3_a4] mode={args.mode} fold={cfg.fold} "
        f"seed={cfg.seed} panel={cfg.panel_kind} "
        f"sector_universe={cfg.pretrain_sector_universe_id} "
        f"comove_universe={cfg.pretrain_comovement_universe_id} "
        f"device={device}",
        flush=True,
    )

    ckpt_dir = Path(cfg.output_dir) / "_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold{cfg.fold}_encoder.pt"

    if not args.skip_pretrain:
        if ckpt_path.exists():
            print(
                f"[universe_layer1_a3_a4] stage 1 ckpt exists at "
                f"{ckpt_path}; skipping pretrain (idempotent).",
                flush=True,
            )
        else:
            _set_seed(42)  # canonical: stage 1 always uses seed 42
            if args.mode == "a3":
                run_stage1_sequential_pretrain(
                    cfg, int(args.pretrain_epochs), device, ckpt_path,
                )
            else:
                run_stage1_pretrain(
                    cfg, int(args.pretrain_epochs), device, ckpt_path,
                )
    if args.pretrain_only:
        return 0

    os.environ["INVAR_SAVE_FULL_STATE"] = "1"
    _set_seed(int(args.seed))
    run_stage2_finetune(
        cfg, int(args.finetune_epochs), device, ckpt_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
