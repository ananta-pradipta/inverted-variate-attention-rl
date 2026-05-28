"""NASDAQ-100 Layer 1 with A1 sequential pretrain regime -> sector.

Per-universe Stage 1 entrypoint for the A1 (2026-05-27) cross-universe
transfer test on the NASDAQ-100 lattice_native-schema panel. Mirrors the
SP500 A1 path that ``invar_rl_sp500_canonical_a1_seq_regime_sector_
stage1.sbatch`` exercises via ``invar_rl.training.stage1_rank``, but
routes through ``cfg.panel_kind = "nasdaq100"``.

This module is ONLY a thin (fold, seed) runner: it builds an
``InvarSTXV2Config`` for NASDAQ-100 with the A1 cfg fields set, then
calls the canonical sequential pretrain + finetune from
``src.baselines.train_invar_clpretrain_v2``. The A1 sequential wrapper
``run_stage1_sequential_pretrain`` is left BYTE-IDENTICAL; this file
does not modify any A1 source code per the canonical no-edit
constraint (commit 837882d, READ-ONLY per the task spec).

Per-fold encoder ckpt lives at
``outputs/nasdaq100/layer1_a1_seq_regime_sector/_ckpt/fold{F}_encoder.pt``
and is shared across the 5 finetune seeds (42..46). The C3 sector
cache (re-used for Stage 1b) lives at
``cache/sector_labels/nasdaq100.parquet`` (built once via
``invar_rl.scripts.build_nasdaq100_sector_map``).

Usage::

    python -m invar_rl.training.nasdaq100_layer1_a1_seq_regime_sector \\
        --fold 1 --seed 42 \\
        --output-dir outputs/nasdaq100/layer1_a1_seq_regime_sector \\
        --pretrain-epochs 10 --finetune-epochs 10 \\
        --pretrain-sector-universe-id nasdaq100

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
    pretrain_sector_universe_id: str,
):
    """Build the canonical NASDAQ-100 InVAR cfg with A1 sequential flags."""
    from src.baselines.train_invar_stx_v2 import InvarSTXV2Config

    cfg = InvarSTXV2Config(fold=int(fold), seed=int(seed))
    cfg.panel_kind = "nasdaq100"
    cfg.two_regime_val = True
    cfg.panel_end = str(panel_end)
    cfg.output_dir = str(output_dir)
    cfg.enable_retrieval_bank = False
    # A1 (2026-05-27): sequential curriculum. Default "regime,sector"
    # runs Stage 1a regime InfoNCE then Stage 1b C3 per-stock same-sector
    # SupCon InfoNCE. The sequential wrapper dispatches off
    # cfg.pretrain_stages and continues training the same encoder
    # weights across stages.
    cfg.pretrain_stages = list(pretrain_stages)
    cfg.pretrain_sector_universe_id = str(pretrain_sector_universe_id)
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "NASDAQ-100 Layer 1 with A1 sequential regime -> sector "
            "pretrain."
        )
    )
    p.add_argument("--fold", type=int, required=True,
                   choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--output-dir", type=str,
        default="outputs/nasdaq100/layer1_a1_seq_regime_sector",
    )
    p.add_argument("--panel-end", type=str, default="2025-12-31")
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument(
        "--pretrain-stages", type=str, default="regime,sector",
        help=(
            "A1 curriculum, comma-separated. Default 'regime,sector'."
        ),
    )
    p.add_argument(
        "--pretrain-sector-universe-id", type=str,
        default="nasdaq100",
        help=(
            "Universe id for the C3 sector cache at "
            "cache/sector_labels/<id>.parquet (default 'nasdaq100')."
        ),
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
        pretrain_sector_universe_id=(
            args.pretrain_sector_universe_id
        ),
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(
        f"[nasdaq100_layer1_a1_seq_regime_sector] fold={cfg.fold} "
        f"seed={cfg.seed} panel={cfg.panel_kind} stages={stages} "
        f"sector_universe_id={cfg.pretrain_sector_universe_id} "
        f"device={device}",
        flush=True,
    )

    ckpt_dir = Path(cfg.output_dir) / "_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold{cfg.fold}_encoder.pt"

    if not args.skip_pretrain:
        if ckpt_path.exists():
            print(
                f"[nasdaq100_layer1_a1_seq_regime_sector] stage 1 ckpt "
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
