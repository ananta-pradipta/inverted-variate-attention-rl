"""Option B Stage-2 driver: per-universe canonical InVAR finetune
starting from a multi-task pretrain backbone.

This driver is a THIN variant of
``invar_rl.training.stage1_rank.run_one_cell`` whose ONLY difference is
the source of the per-fold encoder checkpoint that Stage-2 strict-loads
into ``model.temporal_encoder``:

  * canonical path: ``run_stage1_pretrain`` (single-universe SimCLR
    contrastive pretrain) writes ``foldF_encoder.pt`` to
    ``{output_dir}/_ckpt/`` per the canonical convention.
  * multitask path: ``train_multitask_pretrain`` writes per-universe
    ``foldF_encoder.pt`` files to
    ``{pretrain_dir}/_ckpt_per_universe/{universe}/`` using the SAME
    schema. This driver copies (or symlinks) the chosen universe's file
    to the canonical ``{output_dir}/_ckpt/`` path and then invokes
    ``run_stage2_finetune`` unchanged.

The Stage-2 finetune code path stays BYTE-IDENTICAL to the canonical
``src.baselines.train_invar_clpretrain_v2.run_stage2_finetune`` so the
audit cross-check "canonical paths preserved when multitask flags OFF"
holds: when this driver is not invoked, the canonical pipeline is
exactly the canonical pipeline.

Usage::

    PYTHONPATH=$PWD python3 -m invar_rl.training.multitask_l1_finetune \\
        --universe lattice_native --fold 1 --seed 42 \\
        --multitask-pretrain-dir invar_rl/results/multitask_l1 \\
        --output-dir invar_rl/results/multitask_l1/lattice_native \\
        --finetune-epochs 10
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch

from src.invar.canonical import (
    InVARConfig,
    finetune_invar,
)
from src.models.multitask_l1 import UNIVERSE_FEATURE_DIMS


def _resolve_multitask_ckpt(
    pretrain_dir: Path,
    universe: str,
    fold: int,
) -> Path:
    """Locate the multitask-pretrain encoder ckpt for one (universe, fold).

    Raises FileNotFoundError if missing.
    """
    ckpt_path = (
        pretrain_dir / "_ckpt_per_universe" / universe
        / f"fold{fold}_encoder.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"[ERR] multitask pretrain ckpt not found: {ckpt_path}. "
            f"Run src.baselines.train_multitask_pretrain for "
            f"fold={fold} first."
        )
    return ckpt_path


def _stage_canonical_ckpt(
    multitask_ckpt: Path,
    output_dir: Path,
    fold: int,
) -> Path:
    """Materialise the multitask ckpt at the canonical per-fold path so
    ``run_stage2_finetune`` finds it via its usual lookup.

    Copies (not symlinks) so the canonical ckpt sits inside the
    per-universe output_dir tree and audits cleanly. Idempotent: if the
    destination already exists with the same payload, the copy is
    skipped.
    """
    canonical_ckpt_dir = output_dir / "_ckpt"
    canonical_ckpt_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = canonical_ckpt_dir / f"fold{fold}_encoder.pt"
    if canonical_path.exists():
        # Defensive: verify the staged file is the multitask one (we
        # require an explicit overwrite if there is a canonical file
        # already in place from a prior canonical run).
        ckpt = torch.load(canonical_path, map_location="cpu", weights_only=False)
        if bool(ckpt.get("multitask_pretrain", False)):
            print(
                f"[INFO] canonical ckpt already staged from multitask "
                f"source: {canonical_path}",
                flush=True,
            )
            return canonical_path
        raise RuntimeError(
            f"[ERR] refusing to overwrite non-multitask ckpt at "
            f"{canonical_path}; archive or remove it manually first."
        )
    # Atomic stage: copy to a unique temp path, then os.replace into
    # place. Avoids partial-file races when multiple SLOTS of the
    # finetune sbatch hit the same fold concurrently.
    tmp_path = canonical_path.with_suffix(
        f".tmp.pid{os.getpid()}"
    )
    shutil.copy2(multitask_ckpt, tmp_path)
    os.replace(tmp_path, canonical_path)
    print(
        f"[INFO] staged multitask ckpt -> {canonical_path} "
        f"(from {multitask_ckpt})",
        flush=True,
    )
    return canonical_path


def run_one_cell(
    universe: str,
    fold: int,
    seed: int,
    multitask_pretrain_dir: Path,
    output_dir: Path,
    finetune_epochs: int,
    panel_end: str,
    two_regime_val: bool,
    device: torch.device,
) -> None:
    """Stage-2 finetune for one (universe, fold, seed) cell starting
    from the multi-task pretrained backbone for ``universe``."""
    if universe not in UNIVERSE_FEATURE_DIMS:
        raise ValueError(
            f"[ERR] universe={universe!r} not registered in "
            f"UNIVERSE_FEATURE_DIMS={sorted(UNIVERSE_FEATURE_DIMS)}"
        )

    # Locate + stage the multitask-pretrain ckpt at the canonical path.
    multitask_ckpt = _resolve_multitask_ckpt(
        multitask_pretrain_dir, universe, fold,
    )
    canonical_ckpt = _stage_canonical_ckpt(
        multitask_ckpt, output_dir, fold,
    )

    # Build the canonical InVAR config and route to the per-universe
    # panel via panel_kind == universe id.
    cfg = InVARConfig(fold=fold, seed=seed)
    cfg.panel_kind = universe
    cfg.two_regime_val = bool(two_regime_val)
    cfg.panel_end = str(panel_end)
    cfg.output_dir = str(output_dir)
    # BANKLESS canonical invariant (matches all per-universe sbatches).
    if cfg.enable_retrieval_bank:
        raise RuntimeError(
            "[ERR] canonical InVAR is bankless; enable_retrieval_bank "
            "must be False."
        )

    # Persist the full finetuned state so downstream L2L3 SAC eval can
    # consume foldF_seedS_full.pt; matches every per-universe sbatch
    # under invar_rl/scripts/wulver/.
    os.environ["INVAR_SAVE_FULL_STATE"] = "1"
    print(
        f"[INFO] multitask finetune universe={universe} fold={fold} "
        f"seed={seed} ckpt={canonical_ckpt} device={device} "
        f"epochs={finetune_epochs}",
        flush=True,
    )
    finetune_invar(cfg, finetune_epochs, device, canonical_ckpt)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Option B Stage-2: per-universe canonical InVAR finetune "
            "starting from the multi-task pretrain backbone."
        )
    )
    p.add_argument(
        "--universe", type=str, required=True,
        choices=sorted(UNIVERSE_FEATURE_DIMS),
        help="panel_kind id (lattice_native / nasdaq100 / biotech_nbi_enriched).",
    )
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--multitask-pretrain-dir", type=str, required=True,
        help=(
            "Directory where train_multitask_pretrain wrote per-universe "
            "ckpts (contains _ckpt_per_universe/{universe}/foldF_encoder.pt)."
        ),
    )
    p.add_argument(
        "--output-dir", type=str, required=True,
        help=(
            "Per-universe Stage-2 output root; canonical clpretrain "
            "writes foldF_seedS.json and _ckpt/foldF_seedS_full.pt here."
        ),
    )
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--panel_end", type=str, default="2025-12-31")
    p.add_argument(
        "--two_regime_val", action="store_true", default=True,
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_one_cell(
        universe=args.universe,
        fold=args.fold,
        seed=args.seed,
        multitask_pretrain_dir=Path(args.multitask_pretrain_dir),
        output_dir=Path(args.output_dir),
        finetune_epochs=int(args.finetune_epochs),
        panel_end=str(args.panel_end),
        two_regime_val=bool(args.two_regime_val),
        device=device,
    )


if __name__ == "__main__":
    main()
