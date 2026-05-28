"""Cross-universe Layer 1 DSL (differentiable Sharpe) training driver.

Mirrors :mod:`invar_rl.training.nasdaq100_layer1_eval` and
:mod:`invar_rl.training.biotech_nbi_enriched_layer1_eval` but augments
the InvarSTXV2Config with the Option C differentiable-Sharpe surrogate
attrs (loss_config, diff_sharpe_weight, diff_sharpe_K,
diff_sharpe_temperature, diff_sharpe_batch_days). The DSL term is
consumed inside :func:`src.baselines.train_invar_clpretrain_v2.run_stage2_finetune`
via getattr(cfg, ...) reads, so the only requirement is that the
attrs are set before run_stage2_finetune is called.

This file is NEW (created 2026-05-26), not a modification of the
DSL commit (6f5eaee); the DSL implementation in src/invar/training/loss.py
and src/baselines/train_invar_clpretrain_v2.py is left untouched.

Constraint precedent: per-universe layer1 drivers already set
cfg.panel_kind / cfg.two_regime_val / cfg.panel_end / cfg.output_dir /
cfg.enable_retrieval_bank ; this driver adds 5 more cfg attrs in the
same style.

CLI::

    python -m invar_rl.training.universe_layer1_diff_sharpe_eval \\
        --universe {nasdaq100|biotech_nbi_enriched} \\
        --fold F --seed S \\
        --output-dir-root <path> \\
        --diff-sharpe-k <K_per_universe>

K plumbing: --diff-sharpe-k is universe-specific (K=20 NDX, K=25 NBI,
K=50 SP500) and matches the wrapper top-K cardinality so the soft
Sharpe surrogate is computed on the same portfolio cardinality the
test-time wrapper uses.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


_UNIVERSE_CHOICES = ("nasdaq100", "biotech_nbi_enriched")


def _set_seed(seed: int) -> None:
    """Deterministic seeding for numpy / torch / python random."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_cell(
    universe: str,
    fold: int,
    seed: int,
    output_dir_root: Path,
    pretrain_epochs: int,
    finetune_epochs: int,
    panel_end: str,
    diff_sharpe_weight: float,
    diff_sharpe_k: int,
    diff_sharpe_temperature: float,
    diff_sharpe_batch_days: int,
) -> Dict[str, Any]:
    """Train one (fold, seed) cell of DSL-augmented InVAR on the given universe.

    Args:
        universe: One of ``_UNIVERSE_CHOICES``. Routes panel_kind +
            output paths to the correct lattice-bridge data files.
        fold: 1..5 walk-forward fold index.
        seed: integer finetune seed (42..46 in the DSL sweep).
        output_dir_root: results root; the canonical trainer writes
            ``output_dir_root/fold{F}_seed{S}.json`` and persists the
            per-fold encoder + per-seed full state_dict ckpts under
            ``output_dir_root/_ckpt/``.
        pretrain_epochs: stage-1 InfoNCE epochs (canonical 10).
        finetune_epochs: stage-2 supervised epochs (canonical 10).
        panel_end: ``cfg.panel_end`` (last calendar day of the panel).
        diff_sharpe_weight: weight on the DSL term added to cs_loss
            (default 0.2 per Option C preset).
        diff_sharpe_k: per-side soft top-K wrapper size used inside the
            DSL surrogate (NDX=20, NBI=25, SP500=50).
        diff_sharpe_temperature: sigmoid temperature for the soft top-K
            relaxation (default 0.1).
        diff_sharpe_batch_days: rolling window length in training days
            over which the Sharpe std is computed (default 16).

    Returns:
        Dict with the cell metadata copied from the canonical JSON.
    """
    if universe not in _UNIVERSE_CHOICES:
        raise ValueError(
            f"unsupported universe={universe!r}; "
            f"expected one of {_UNIVERSE_CHOICES}"
        )

    from src.baselines.train_invar_clpretrain_v2 import (
        run_stage1_pretrain, run_stage2_finetune,
    )
    from src.baselines.train_invar_stx_v2 import InvarSTXV2Config

    cfg = InvarSTXV2Config(fold=fold, seed=seed)
    cfg.panel_kind = universe
    cfg.two_regime_val = True
    cfg.panel_end = panel_end
    cfg.output_dir = str(output_dir_root)
    cfg.enable_retrieval_bank = False

    # Option C (DSL) attrs. The trainer reads these via getattr() with
    # safe defaults; setting them here activates the DSL term inside
    # run_stage2_finetune.
    cfg.loss_config = "diff_sharpe"
    cfg.diff_sharpe_weight = float(diff_sharpe_weight)
    cfg.diff_sharpe_K = int(diff_sharpe_k)
    cfg.diff_sharpe_temperature = float(diff_sharpe_temperature)
    cfg.diff_sharpe_batch_days = int(diff_sharpe_batch_days)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[universe_layer1_dsl universe={universe}] fold={fold} seed={seed} "
        f"device={device} K={diff_sharpe_k} tau={diff_sharpe_temperature} "
        f"weight={diff_sharpe_weight} batch_days={diff_sharpe_batch_days} "
        f"pretrain_epochs={pretrain_epochs} finetune_epochs={finetune_epochs}",
        flush=True,
    )

    ckpt_dir = Path(cfg.output_dir) / "_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold{fold}_encoder.pt"

    # Stage 1 (pretrain) is shared across seeds for the same fold;
    # canonical protocol uses seed=42 for the pretrain.
    if not ckpt_path.exists():
        _set_seed(42)
        run_stage1_pretrain(cfg, pretrain_epochs, device, ckpt_path)
    else:
        print(
            f"[universe_layer1_dsl universe={universe}] stage 1 ckpt "
            f"exists; reusing {ckpt_path}",
            flush=True,
        )

    # Persist the full finetuned state_dict alongside the JSON so the
    # downstream Layer 3 SAC driver can load it.
    os.environ["INVAR_SAVE_FULL_STATE"] = "1"
    _set_seed(seed)
    run_stage2_finetune(cfg, finetune_epochs, device, ckpt_path)

    canonical_json = (
        Path(cfg.output_dir) / f"fold{fold}_seed{seed}.json"
    )
    with open(canonical_json) as fh:
        payload = json.load(fh)
    return payload


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Cross-universe Layer 1 training with the DSL (differentiable "
            "Sharpe) loss augmentation. Supports NDX-100 and NBI-enriched."
        )
    )
    p.add_argument("--universe", type=str, required=True,
                   choices=list(_UNIVERSE_CHOICES))
    p.add_argument("--fold", type=int, required=True,
                   choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir-root", type=str, required=True)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--panel-end", type=str, default="2025-12-31")
    p.add_argument("--diff-sharpe-weight", type=float, default=0.2)
    p.add_argument("--diff-sharpe-k", type=int, required=True,
                   help=(
                       "Per-side soft top-K wrapper size for the DSL "
                       "surrogate. Pass 20 for NDX-100, 25 for "
                       "NBI-enriched, 50 for SP500 (SP500 has its own "
                       "sbatch in stage1_diff_sharpe_phase2)."
                   ))
    p.add_argument("--diff-sharpe-temperature", type=float, default=0.1)
    p.add_argument("--diff-sharpe-batch-days", type=int, default=16)
    args = p.parse_args()

    output_dir_root = Path(args.output_dir_root)
    output_dir_root.mkdir(parents=True, exist_ok=True)

    # Skip if both the per-seed full ckpt AND the per-cell JSON exist.
    full_ckpt = output_dir_root / "_ckpt" / (
        f"fold{args.fold}_seed{args.seed}_full.pt"
    )
    cell_json = output_dir_root / f"fold{args.fold}_seed{args.seed}.json"
    if full_ckpt.exists() and cell_json.exists():
        print(
            f"[universe_layer1_dsl universe={args.universe}] "
            f"{full_ckpt.name} + {cell_json.name} already exist; "
            f"skipping cell",
            flush=True,
        )
        return 0

    _train_cell(
        universe=args.universe,
        fold=args.fold,
        seed=args.seed,
        output_dir_root=output_dir_root,
        pretrain_epochs=int(args.pretrain_epochs),
        finetune_epochs=int(args.finetune_epochs),
        panel_end=args.panel_end,
        diff_sharpe_weight=float(args.diff_sharpe_weight),
        diff_sharpe_k=int(args.diff_sharpe_k),
        diff_sharpe_temperature=float(args.diff_sharpe_temperature),
        diff_sharpe_batch_days=int(args.diff_sharpe_batch_days),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
