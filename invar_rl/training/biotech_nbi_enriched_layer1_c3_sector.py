"""biotech NBI-enriched Layer 1 with C3 (sector-aware positives) selector.

Per-universe Stage 1 entrypoint for the C3 (2026-05-27) cross-universe
transfer test on the biotech NBI-enriched 22-feature panel using
SUB-INDUSTRY granularity (not the degenerate top-level GICS "Health
Care" cohort that caused the original C3 NBI rollup to SKIP).

Mirrors the NDX C3 path
(``invar_rl.training.nasdaq100_layer1_c3_sector``) but routes through
``cfg.panel_kind = "biotech_nbi_enriched"`` and pulls the sector cache
from ``cache/sector_labels/biotech_nbi_enriched.parquet`` (built once
via ``invar_rl.scripts.build_biotech_nbi_enriched_sector_map``).

The ``sector_id`` column in the NBI cache encodes 9 healthcare-focused
SUB-INDUSTRY cohorts (Biotechnology, Pharmaceuticals, Pharmaceuticals
Generic, Health Care Equipment, Health Care Supplies, Life Sciences
Tools Svc, Health Care Technology, Health Care Providers Svc, Health
Care Distributors). The C3 selector is universe-agnostic: it only
compares the int sector_id, so the same hook works for top-level
sectors (SP500 / NDX) and sub-industries (NBI) without source-code
changes.

This module is ONLY a thin (fold, seed) runner: it builds an
``InvarSTXV2Config`` for biotech_nbi_enriched with the C3 cfg fields
set, then calls the canonical pretrain + finetune from
``src.baselines.train_invar_clpretrain_v2``. The C3 selector itself
(``src.models.pretrain_improvements.sector_positives`` + the
``pretrain_positive_method == "sector"`` branch in
``run_stage1_pretrain``) is left BYTE-IDENTICAL; this file does not
modify any C3 source code per the canonical no-edit constraint.

Per-fold encoder ckpt lives at
``outputs/biotech_nbi_enriched/layer1_c3_sector/_ckpt/fold{F}_encoder.pt``
and is shared across the 5 finetune seeds (42..46).

Usage::

    python -m invar_rl.training.biotech_nbi_enriched_layer1_c3_sector \\
        --fold 1 --seed 42 \\
        --output-dir outputs/biotech_nbi_enriched/layer1_c3_sector \\
        --pretrain-epochs 10 --finetune-epochs 10 \\
        --pretrain-sector-universe-id biotech_nbi_enriched

Add ``--pretrain-only`` to run Stage 1.a (per-fold pretrain) and exit
without touching Stage 1.b; add ``--skip-pretrain`` to skip Stage 1.a
and load an existing per-fold encoder ckpt for Stage 1.b.
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
    pretrain_sector_universe_id: str,
):
    """Build the canonical biotech_nbi_enriched InVAR cfg with C3 flags."""
    from src.baselines.train_invar_stx_v2 import InvarSTXV2Config

    cfg = InvarSTXV2Config(fold=int(fold), seed=int(seed))
    cfg.panel_kind = "biotech_nbi_enriched"
    cfg.two_regime_val = True
    cfg.panel_end = str(panel_end)
    cfg.output_dir = str(output_dir)
    cfg.enable_retrieval_bank = False
    # C3 (2026-05-27): switch the Stage-1 contrastive positive selector
    # from kmeans (day-level L2-NN over the 14-d episode-key
    # fingerprint) to the per-day per-stock SupCon InfoNCE where
    # positives are same-day same-sector peers (negatives = same-day
    # different-sector peers). On the NBI universe the
    # ``pretrain_sector_universe_id`` resolves to a SUB-INDUSTRY
    # parquet (~73% Biotechnology + 6 minority cohorts), avoiding the
    # top-level-sector degeneracy that originally caused the C3 NBI
    # rollup to SKIP.
    cfg.pretrain_positive_method = "sector"
    cfg.pretrain_sector_universe_id = str(pretrain_sector_universe_id)
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "biotech NBI-enriched Layer 1 with C3 sector-aware "
            "positives pretrain (sub-industry granularity)."
        )
    )
    p.add_argument("--fold", type=int, required=True,
                   choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--output-dir", type=str,
        default="outputs/biotech_nbi_enriched/layer1_c3_sector",
    )
    p.add_argument("--panel-end", type=str, default="2025-12-31")
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--pretrain-only", action="store_true",
                   help="Run only Stage 1.a (per-fold pretrain).")
    p.add_argument("--skip-pretrain", action="store_true",
                   help="Skip Stage 1.a; load existing per-fold ckpt.")
    p.add_argument(
        "--pretrain-sector-universe-id", type=str,
        default="biotech_nbi_enriched",
        help=(
            "Universe id for the C3 sector cache at "
            "cache/sector_labels/<id>.parquet (default "
            "'biotech_nbi_enriched')."
        ),
    )
    args = p.parse_args()

    if args.pretrain_only and args.skip_pretrain:
        raise SystemExit(
            "--pretrain-only and --skip-pretrain are mutually exclusive."
        )

    from src.baselines.train_invar_clpretrain_v2 import (
        run_stage1_pretrain, run_stage2_finetune,
    )

    cfg = _build_cfg(
        fold=args.fold,
        seed=args.seed,
        output_dir=args.output_dir,
        panel_end=args.panel_end,
        pretrain_sector_universe_id=args.pretrain_sector_universe_id,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(
        f"[biotech_nbi_enriched_layer1_c3_sector] fold={cfg.fold} "
        f"seed={cfg.seed} panel={cfg.panel_kind} "
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
                f"[biotech_nbi_enriched_layer1_c3_sector] stage 1 ckpt "
                f"exists at {ckpt_path}; skipping pretrain (idempotent).",
                flush=True,
            )
        else:
            _set_seed(42)  # canonical: stage 1 always uses seed 42
            run_stage1_pretrain(
                cfg, int(args.pretrain_epochs), device, ckpt_path
            )
    if args.pretrain_only:
        return 0

    # Stage 1.b: per-seed finetune. INVAR_SAVE_FULL_STATE=1 ensures the
    # full state_dict is persisted alongside the JSON so Stage 2/3 SAC
    # can re-load the trained model.
    os.environ["INVAR_SAVE_FULL_STATE"] = "1"
    _set_seed(int(args.seed))
    run_stage2_finetune(
        cfg, int(args.finetune_epochs), device, ckpt_path
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
