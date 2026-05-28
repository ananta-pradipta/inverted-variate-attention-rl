"""NASDAQ-100 Layer 1 with A2 sequential pretrain regime -> co-movement.

Per-universe Stage 1 entrypoint for the A2 (2026-05-27) cross-universe
transfer test on the NASDAQ-100 lattice_native-schema panel. Mirrors the
SP500 A2 path that ``invar_rl_sp500_canonical_a2_seq_regime_comove_
stage1.sbatch`` exercises via ``invar_rl.training.stage1_rank``, but
routes through ``cfg.panel_kind = "nasdaq100"``.

This module is ONLY a thin (fold, seed) runner: it builds an
``InvarSTXV2Config`` for NASDAQ-100 with the A2 cfg fields set, then
calls the canonical sequential pretrain + finetune from
``src.baselines.train_invar_clpretrain_v2``. The A2 sequential wrapper
and the comovement clusterer are left BYTE-IDENTICAL; this file does
not modify any A2 source code per the canonical no-edit constraint.

Per-fold encoder ckpt lives at
``outputs/nasdaq100/layer1_a2_seq_regime_comove/_ckpt/fold{F}_encoder.pt``
and is shared across the 5 finetune seeds (42..46). The A2 cluster cache
lives at
``cache/pretrain_improvements/comovement/nasdaq100/foldF/cluster_ids.parquet``
and is pre-built once via
``invar_rl.scripts.build_comovement_clusters --panel-kind nasdaq100 ...``
(or rebuilt JIT inside ``run_stage1_pretrain`` on cache miss).

Usage::

    python -m invar_rl.training.nasdaq100_layer1_a2_seq_regime_comove \\
        --fold 1 --seed 42 \\
        --output-dir outputs/nasdaq100/layer1_a2_seq_regime_comove \\
        --pretrain-epochs 10 --finetune-epochs 10 \\
        --pretrain-comovement-universe-id nasdaq100 \\
        --pretrain-comovement-n-clusters 8 \\
        --pretrain-comovement-window 252 \\
        --pretrain-comovement-epochs 5

Add ``--pretrain-only`` to run Stage 1.a + 1.b (per-fold pretrain) and
exit without touching Stage 2; add ``--skip-pretrain`` to skip pretrain
and load an existing per-fold encoder ckpt for Stage 2.
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


def _build_cfg(
    fold: int,
    seed: int,
    output_dir: str,
    panel_end: str,
    pretrain_stages: list[str],
    pretrain_comovement_universe_id: str,
    pretrain_comovement_n_clusters: int,
    pretrain_comovement_window: int,
    pretrain_comovement_epochs: int,
):
    """Build the canonical NASDAQ-100 InVAR cfg with A2 sequential flags."""
    from src.baselines.train_invar_stx_v2 import InvarSTXV2Config

    cfg = InvarSTXV2Config(fold=int(fold), seed=int(seed))
    cfg.panel_kind = "nasdaq100"
    cfg.two_regime_val = True
    cfg.panel_end = str(panel_end)
    cfg.output_dir = str(output_dir)
    cfg.enable_retrieval_bank = False
    # A2 (2026-05-27): sequential curriculum. Default "regime,comovement"
    # runs Stage 1a regime InfoNCE then Stage 1b co-movement-clustered
    # per-stock SupCon InfoNCE.
    cfg.pretrain_stages = list(pretrain_stages)
    cfg.pretrain_comovement_universe_id = str(
        pretrain_comovement_universe_id
    )
    cfg.pretrain_comovement_n_clusters = int(
        pretrain_comovement_n_clusters
    )
    cfg.pretrain_comovement_window = int(pretrain_comovement_window)
    cfg.pretrain_comovement_epochs = int(pretrain_comovement_epochs)
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "NASDAQ-100 Layer 1 with A2 sequential regime -> co-movement "
            "pretrain."
        )
    )
    p.add_argument("--fold", type=int, required=True,
                   choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--output-dir", type=str,
        default="outputs/nasdaq100/layer1_a2_seq_regime_comove",
    )
    p.add_argument("--panel-end", type=str, default="2025-12-31")
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument(
        "--pretrain-stages", type=str, default="regime,comovement",
        help=(
            "A2 curriculum, comma-separated. Default 'regime,comovement'."
        ),
    )
    p.add_argument(
        "--pretrain-comovement-universe-id", type=str,
        default="nasdaq100",
        help=(
            "Cache key under cache/pretrain_improvements/comovement/"
            "<id>/foldF/cluster_ids.parquet (default 'nasdaq100')."
        ),
    )
    p.add_argument(
        "--pretrain-comovement-n-clusters", type=int, default=8,
    )
    p.add_argument(
        "--pretrain-comovement-window", type=int, default=252,
    )
    p.add_argument(
        "--pretrain-comovement-epochs", type=int, default=5,
        help="Stage 1b epoch count (default 5).",
    )
    p.add_argument("--pretrain-only", action="store_true",
                   help="Run only Stage 1 (per-fold pretrain).")
    p.add_argument("--skip-pretrain", action="store_true",
                   help="Skip Stage 1; load existing per-fold ckpt.")
    args = p.parse_args()

    if args.pretrain_only and args.skip_pretrain:
        raise SystemExit(
            "--pretrain-only and --skip-pretrain are mutually exclusive."
        )

    from src.baselines.train_invar_clpretrain_v2 import (
        run_stage1_sequential_pretrain,
        run_stage2_finetune,
    )

    stages = [
        s.strip() for s in str(args.pretrain_stages).split(",") if s.strip()
    ]
    cfg = _build_cfg(
        fold=args.fold,
        seed=args.seed,
        output_dir=args.output_dir,
        panel_end=args.panel_end,
        pretrain_stages=stages,
        pretrain_comovement_universe_id=(
            args.pretrain_comovement_universe_id
        ),
        pretrain_comovement_n_clusters=(
            args.pretrain_comovement_n_clusters
        ),
        pretrain_comovement_window=args.pretrain_comovement_window,
        pretrain_comovement_epochs=args.pretrain_comovement_epochs,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(
        f"[nasdaq100_layer1_a2_seq_regime_comove] fold={cfg.fold} "
        f"seed={cfg.seed} panel={cfg.panel_kind} stages={stages} "
        f"comove_universe={cfg.pretrain_comovement_universe_id} "
        f"K={cfg.pretrain_comovement_n_clusters} "
        f"window={cfg.pretrain_comovement_window} "
        f"comove_epochs={cfg.pretrain_comovement_epochs} "
        f"device={device}",
        flush=True,
    )

    ckpt_dir = Path(cfg.output_dir) / "_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold{cfg.fold}_encoder.pt"

    if not args.skip_pretrain:
        if ckpt_path.exists():
            print(
                f"[nasdaq100_layer1_a2_seq_regime_comove] stage 1 ckpt "
                f"exists at {ckpt_path}; skipping pretrain (idempotent).",
                flush=True,
            )
        else:
            _set_seed(42)  # canonical: stage 1 always uses seed 42
            run_stage1_sequential_pretrain(
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
