"""NASDAQ-100 Phase 3 Layer 1 (canonical InVAR) training driver.

Thin wrapper around the canonical bankless + clpretrain InVAR pipeline
in :mod:`src.baselines.train_invar_clpretrain_v2` for a single
``(fold, seed)`` cell on the NASDAQ-100 universe.

Constraints (per the Phase 3 spec):
- Reuses the canonical InVAR backbone architecture byte-for-byte.
- Hyperparameters held byte-for-byte from the S&P 500 protocol (no
  retuning): 10 stage-1 InfoNCE pretrain epochs, 10 stage-2 supervised
  finetune epochs, ``two_regime_val=True`` (fixed val = 2017 H2 + 2018 H2).
- panel_kind = ``nasdaq100``; the v2 runner routes panel / mask / macro /
  betas paths to the NASDAQ-100 parquets (see Phase 2 report and the
  ``panel_kind == "nasdaq100"`` branches in
  ``src.baselines.v2_runner.build_panel`` / ``build_masks`` /
  ``fold_split``, and in ``src.baselines.train_invar_stx_v2`` /
  ``src.baselines.train_invar_clpretrain_v2`` for the macro / betas
  feeds.

Per cell, after training, this driver additionally persists:
- daily scores at outputs/nasdaq100/layer1/scores/fold{F}_seed{S}.parquet
  (one row per test (day, ticker), float32 score)
- macro encodings at outputs/nasdaq100/layer1/macro_enc/fold{F}_seed{S}.parquet
  (one row per test day, float32 macro encoder output of dim
  ``cfg.macro_out_dim``)
- per-cell metadata JSON at outputs/nasdaq100/layer1/metrics/fold{F}_seed{S}.json
  (fold, seed, val_rank_ic, test_rank_ic, test_ic, test_ndcg10/50, etc.).

The canonical clpretrain trainer also writes its own JSON at
``cfg.output_dir / fold{F}_seed{S}.json``; we read that file and copy
the headline numbers into the cell metadata JSON so downstream rollups
have a single canonical schema.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def _set_seed(seed: int) -> None:
    """Deterministic seeding for numpy / torch / python random."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_cell(
    fold: int,
    seed: int,
    output_dir_root: Path,
    pretrain_epochs: int,
    finetune_epochs: int,
    panel_end: str,
    smoke: bool,
) -> Dict[str, Any]:
    """Train one (fold, seed) cell of canonical InVAR on NASDAQ-100.

    Stage 1 (per-fold contrastive pretrain) is shared across seeds for
    that fold via the ``foldF_encoder.pt`` ckpt convention; this driver
    runs it once per cell to keep skip-if-exists semantics clean (the
    canonical trainer guards against re-running stage 1 if the ckpt
    already exists on disk via the ``--skip_pretrain`` flag, but we
    always run both stages here since the cost is small and the per-cell
    wall-time is dominated by stage 2).

    Args:
        fold: 1..5 walk-forward fold index.
        seed: integer finetune seed (42..46 in the Phase 3 sweep).
        output_dir_root: results root; the canonical trainer writes
            ``output_dir_root/fold{F}_seed{S}.json`` and persists the
            per-fold encoder ckpt under ``output_dir_root/_ckpt/``.
        pretrain_epochs: stage-1 epochs (default 10 in canonical protocol).
        finetune_epochs: stage-2 epochs (default 10 in canonical protocol).
        panel_end: ``cfg.panel_end`` (last calendar day of the panel).
        smoke: if True, run 1 epoch of each stage for fast smoke checks.

    Returns:
        Dict with the cell metadata copied from the canonical JSON.
    """
    from src.baselines.train_invar_clpretrain_v2 import (
        run_stage1_pretrain, run_stage2_finetune,
    )
    from src.baselines.train_invar_stx_v2 import InvarSTXV2Config

    if smoke:
        pretrain_epochs = 1
        finetune_epochs = 1

    cfg = InvarSTXV2Config(fold=fold, seed=seed)
    cfg.panel_kind = "nasdaq100"
    cfg.two_regime_val = True
    cfg.panel_end = panel_end
    cfg.output_dir = str(output_dir_root)
    # BANKLESS canonical invariant. Pretrain + finetune both assert this.
    cfg.enable_retrieval_bank = False
    if smoke and cfg.swa_warmup_epochs >= finetune_epochs:
        cfg.swa_warmup_epochs = max(0, finetune_epochs - 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[nasdaq100_layer1] fold={fold} seed={seed} device={device} "
        f"pretrain_epochs={pretrain_epochs} finetune_epochs={finetune_epochs}",
        flush=True,
    )

    ckpt_dir = Path(cfg.output_dir) / "_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold{fold}_encoder.pt"

    if not ckpt_path.exists():
        _set_seed(42)  # stage 1 always uses seed 42 in canonical protocol
        run_stage1_pretrain(cfg, pretrain_epochs, device, ckpt_path)
    else:
        print(
            f"[nasdaq100_layer1] stage 1 ckpt exists; reusing {ckpt_path}",
            flush=True,
        )

    # Persist the full finetuned state_dict alongside the JSON so the
    # downstream score/macro-enc persistence can re-load the trained
    # model without re-running stage 2.
    os.environ["INVAR_SAVE_FULL_STATE"] = "1"
    _set_seed(seed)
    run_stage2_finetune(cfg, finetune_epochs, device, ckpt_path)

    canonical_json = (
        Path(cfg.output_dir) / f"fold{fold}_seed{seed}.json"
    )
    with open(canonical_json) as fh:
        payload = json.load(fh)
    return payload


def _persist_scores_and_macro(
    fold: int,
    seed: int,
    output_dir_root: Path,
    scores_dir: Path,
    macro_dir: Path,
    panel_end: str,
) -> None:
    """Re-run forward over the test split to persist daily scores +
    macro encodings.

    Loads the full finetuned ``cfg.output_dir/_ckpt/fold{F}_seed{S}_full.pt``
    saved by ``run_stage2_finetune`` (INVAR_SAVE_FULL_STATE=1 set above)
    and forwards the model over each test day to capture per-(day, ticker)
    scores and the macro encoder's per-day output.
    """
    import pandas as pd
    from src.baselines.train_invar_stx_v2 import (
        InvarSTXModel, InvarSTXV2Config,
    )
    from invar_rl.data.lattice_bridge import build_lattice_bridge

    cfg = InvarSTXV2Config(fold=fold, seed=seed)
    cfg.panel_kind = "nasdaq100"
    cfg.two_regime_val = True
    cfg.panel_end = panel_end
    cfg.output_dir = str(output_dir_root)
    cfg.enable_retrieval_bank = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bridge = build_lattice_bridge(cfg, device=device)

    full_path = (
        Path(cfg.output_dir) / "_ckpt" / f"fold{fold}_seed{seed}_full.pt"
    )
    if not full_path.exists():
        print(
            f"[nasdaq100_layer1] WARN no full ckpt at {full_path}; "
            f"skipping scores+macro persistence",
            flush=True,
        )
        return
    ckpt = torch.load(full_path, map_location=device)

    # day_value_dim is not saved explicitly; infer from the day_values
    # tensor computed for this bridge (it depends only on the panel
    # shape, not the training run).
    cfg.day_value_dim = int(bridge.day_values.shape[1])

    model = InvarSTXModel(
        cfg,
        n_features=int(ckpt["n_features"]),
        day_key_dim=int(ckpt["day_key_dim"]),
        duration_input_dim=int(ckpt["duration_input_dim"]),
        macro_input_dim=int(ckpt["macro_input_dim"]),
        macro_gate_in_dim=int(ckpt["macro_gate_in_dim"]),
    ).to(device)
    model.day_memory.populate(
        keys=bridge.day_keys, values=bridge.day_values,
        day_indices=np.arange(len(bridge.dates)),
        train_day_indices=bridge.train_idx,
    )
    model.day_memory.to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    score_rows = []
    macro_rows = []

    macro_enc_module = getattr(model, "macro_encoder", None)
    macro_out_dim: Optional[int] = None
    macro_captured: Dict[int, np.ndarray] = {}
    hook_handle = None
    if macro_enc_module is not None:
        def _capture(mod, args, output):
            # MacroStateEncoder.forward returns (m_state, m_state) where
            # m_state has shape (macro_out_dim,). Capture the first element.
            t = output[0] if isinstance(output, tuple) else output
            macro_captured["latest"] = (
                t.detach().float().cpu().numpy().astype(np.float32)
            )
        hook_handle = macro_enc_module.register_forward_hook(_capture)

    with torch.no_grad():
        for t in bridge.test_idx:
            t = int(t)
            try:
                inp = bridge.day_inputs(t)
            except (ValueError, RuntimeError):
                continue
            active = inp["active_indices"].cpu().numpy().astype(np.int64)
            x_window = inp["x_window"].to(device)
            day_query_key = inp["day_query_key"].to(device)
            allowed = inp["allowed_day_indices"].to(device)
            regime_scalars = inp["regime_scalars"].to(device)
            duration_input = inp["duration_input"].to(device)
            macro_input = inp["macro_input"].to(device)
            macro_gate_input = inp["macro_gate_input"].to(device)

            y_active = model(
                x_window,
                day_query_key=day_query_key,
                query_day_idx=t,
                allowed_day_indices=allowed,
                regime_scalars=regime_scalars,
                duration_input=duration_input,
                macro_input=macro_input,
                macro_gate_input=macro_gate_input,
            ).detach().float().cpu().numpy().astype(np.float32)

            date_str = str(bridge.dates[t])
            for j, n_idx in enumerate(active):
                score_rows.append({
                    "date": date_str,
                    "ticker": bridge.tickers[int(n_idx)],
                    "score": float(y_active[j]),
                })
            if macro_enc_module is not None and "latest" in macro_captured:
                enc = macro_captured["latest"]
                if macro_out_dim is None:
                    macro_out_dim = int(enc.shape[-1])
                macro_rows.append({
                    "date": date_str,
                    **{f"macro_enc_{k}": float(enc[k])
                       for k in range(int(enc.shape[-1]))},
                })
    if hook_handle is not None:
        hook_handle.remove()

    scores_dir.mkdir(parents=True, exist_ok=True)
    macro_dir.mkdir(parents=True, exist_ok=True)
    scores_path = scores_dir / f"fold{fold}_seed{seed}.parquet"
    macro_path = macro_dir / f"fold{fold}_seed{seed}.parquet"
    if score_rows:
        sdf = pd.DataFrame(score_rows)
        sdf["date"] = pd.to_datetime(sdf["date"]).dt.normalize()
        sdf.to_parquet(scores_path, index=False)
        print(
            f"[nasdaq100_layer1] wrote {scores_path}: "
            f"{len(sdf):,} rows ({sdf['date'].nunique()} days x "
            f"{sdf['ticker'].nunique()} tickers)",
            flush=True,
        )
    if macro_rows:
        mdf = pd.DataFrame(macro_rows)
        mdf["date"] = pd.to_datetime(mdf["date"]).dt.normalize()
        mdf.to_parquet(macro_path, index=False)
        print(
            f"[nasdaq100_layer1] wrote {macro_path}: "
            f"{len(mdf):,} days x {macro_out_dim} dim macro_enc",
            flush=True,
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", type=str, default="nasdaq100",
                   choices=["nasdaq100"])
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir-root", type=str,
                   default="outputs/nasdaq100/layer1")
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--panel-end", type=str, default="2025-12-31")
    p.add_argument("--smoke", type=int, default=0,
                   help="1 = run 1 epoch of each stage (smoke check).")
    p.add_argument("--skip-scores", action="store_true",
                   help="Skip the per-day score / macro-enc persistence.")
    args = p.parse_args()

    output_dir_root = Path(args.output_dir_root)
    output_dir_root.mkdir(parents=True, exist_ok=True)

    metrics_dir = output_dir_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"fold{args.fold}_seed{args.seed}.json"

    if metrics_path.exists():
        print(
            f"[nasdaq100_layer1] {metrics_path} already exists; "
            f"skipping cell",
            flush=True,
        )
        return 0

    payload = _train_cell(
        fold=args.fold, seed=args.seed,
        output_dir_root=output_dir_root,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        panel_end=args.panel_end,
        smoke=bool(args.smoke),
    )

    cell_meta = {
        "universe": args.universe,
        "fold": args.fold,
        "seed": args.seed,
        "panel_T": payload["panel_T"],
        "panel_N": payload["panel_N"],
        "panel_F": payload["panel_F"],
        "n_train": payload["n_train"],
        "n_val": payload["n_val"],
        "n_test": payload["n_test"],
        "val_ic": payload["val_ic"],
        "val_rank_ic": payload["val_rank_ic"],
        "test_ic": payload["ic"],
        "test_rank_ic": payload["rank_ic"],
        "test_ndcg10": payload["ndcg10"],
        "test_ndcg50": payload["ndcg50"],
        "test_cohort_ic": payload["test_cohort_ic"],
        "val_cohort_ic": payload["val_cohort_ic"],
        "model": payload["model"],
        "config": payload["config"],
    }
    with open(metrics_path, "w") as fh:
        json.dump(cell_meta, fh, indent=2, default=str)
    print(
        f"[nasdaq100_layer1] wrote {metrics_path}: "
        f"val_rank_ic={cell_meta['val_rank_ic']:+.4f} "
        f"test_rank_ic={cell_meta['test_rank_ic']:+.4f}",
        flush=True,
    )

    if not args.skip_scores:
        scores_dir = output_dir_root / "scores"
        macro_dir = output_dir_root / "macro_enc"
        _persist_scores_and_macro(
            fold=args.fold, seed=args.seed,
            output_dir_root=output_dir_root,
            scores_dir=scores_dir, macro_dir=macro_dir,
            panel_end=args.panel_end,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
