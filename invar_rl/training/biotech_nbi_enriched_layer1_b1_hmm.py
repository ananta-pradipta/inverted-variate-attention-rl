"""biotech NBI-enriched Layer 1 with B1 (HMM regime pretrain) selector.

Per-universe Stage 1 entrypoint for the B1 (2026-05-27) cross-universe
transfer test on the biotech NBI-enriched 22-feature panel. Mirrors the
SP500 B1 path that the canonical SP500 sbatch
``invar_rl_sp500_canonical_b1_hmm_stage1.sbatch`` exercises via
``invar_rl.training.stage1_rank``, but routes through
``cfg.panel_kind = "biotech_nbi_enriched"``.

This module is ONLY a thin (fold, seed) runner: it builds an
``InvarSTXV2Config`` for biotech_nbi_enriched with the B1 HMM cfg fields
set, then calls the canonical pretrain + finetune from
``src.baselines.train_invar_clpretrain_v2``. The B1 HMM hook itself
(``src.models.pretrain_improvements.hmm_regime`` + the
``pretrain_regime_method == "hmm"`` branch in ``run_stage1_pretrain``)
is left BYTE-IDENTICAL; this file does not modify any B1 code.

Per-fold encoder ckpt lives at
``outputs/biotech_nbi_enriched/layer1_b1_hmm/_ckpt/fold{F}_encoder.pt``
and is shared across the 5 finetune seeds (42..46). The HMM posterior
cache lives at ``cache/pretrain_improvements/hmm_regime/
biotech_nbi_enriched/fold{F}/posteriors.parquet``.

Usage::

    python -m invar_rl.training.biotech_nbi_enriched_layer1_b1_hmm \
        --fold 1 --seed 42 \
        --output-dir outputs/biotech_nbi_enriched/layer1_b1_hmm \
        --pretrain-epochs 10 --finetune-epochs 10 \
        --pretrain-hmm-n-states 4 \
        --pretrain-hmm-positive-threshold 0.7 \
        --pretrain-hmm-universe-id biotech_nbi_enriched

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
    pretrain_hmm_n_states: int,
    pretrain_hmm_positive_threshold: float,
    pretrain_hmm_universe_id: str,
):
    """Build the canonical biotech_nbi_enriched InVAR cfg with B1 HMM
    flags set."""
    from src.baselines.train_invar_stx_v2 import InvarSTXV2Config

    cfg = InvarSTXV2Config(fold=int(fold), seed=int(seed))
    cfg.panel_kind = "biotech_nbi_enriched"
    cfg.two_regime_val = True
    cfg.panel_end = str(panel_end)
    cfg.output_dir = str(output_dir)
    cfg.enable_retrieval_bank = False
    # B1 (2026-05-27): switch the Stage-1 contrastive positive selector
    # from kmeans (L2 NN over the 14-d standardised episode-key
    # fingerprint) to hmm (Gaussian HMM posterior cosine similarity over
    # the SAME fingerprint).
    cfg.pretrain_regime_method = "hmm"
    cfg.pretrain_hmm_n_states = int(pretrain_hmm_n_states)
    cfg.pretrain_hmm_positive_threshold = float(
        pretrain_hmm_positive_threshold
    )
    cfg.pretrain_hmm_universe_id = str(pretrain_hmm_universe_id)
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "biotech NBI-enriched Layer 1 with B1 HMM regime pretrain "
            "selector."
        )
    )
    p.add_argument("--fold", type=int, required=True,
                   choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--output-dir", type=str,
        default="outputs/biotech_nbi_enriched/layer1_b1_hmm",
    )
    p.add_argument("--panel-end", type=str, default="2025-12-31")
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--pretrain-only", action="store_true",
                   help="Run only Stage 1.a (per-fold pretrain).")
    p.add_argument("--skip-pretrain", action="store_true",
                   help="Skip Stage 1.a; load existing per-fold ckpt.")
    p.add_argument(
        "--pretrain-hmm-n-states", type=int, default=4,
        help="Number of latent regimes for the B1 HMM (default 4).",
    )
    p.add_argument(
        "--pretrain-hmm-positive-threshold", type=float, default=0.7,
        help=(
            "Cosine-similarity floor on HMM posteriors for B1 SupCon "
            "positives (default 0.7)."
        ),
    )
    p.add_argument(
        "--pretrain-hmm-universe-id", type=str,
        default="biotech_nbi_enriched",
        help=(
            "Universe id for the B1 posterior cache root at "
            "cache/pretrain_improvements/hmm_regime/<id>/foldF/."
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
        pretrain_hmm_n_states=args.pretrain_hmm_n_states,
        pretrain_hmm_positive_threshold=(
            args.pretrain_hmm_positive_threshold
        ),
        pretrain_hmm_universe_id=args.pretrain_hmm_universe_id,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(
        f"[biotech_nbi_enriched_layer1_b1_hmm] fold={cfg.fold} "
        f"seed={cfg.seed} panel={cfg.panel_kind} "
        f"hmm_n_states={cfg.pretrain_hmm_n_states} "
        f"hmm_threshold={cfg.pretrain_hmm_positive_threshold:.2f} "
        f"hmm_universe_id={cfg.pretrain_hmm_universe_id} device={device}",
        flush=True,
    )

    ckpt_dir = Path(cfg.output_dir) / "_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold{cfg.fold}_encoder.pt"

    if not args.skip_pretrain:
        if ckpt_path.exists():
            print(
                f"[biotech_nbi_enriched_layer1_b1_hmm] stage 1 ckpt "
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
