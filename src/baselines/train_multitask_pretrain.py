"""Option B Stage-1: multi-task contrastive pretrain of the canonical
``PerTickerTemporalEncoder`` backbone JOINTLY across multiple universes.

EXPERIMENT-ONLY (never in any paper).

Design
------
Two-stage InVAR canonical training cleanly decomposes into:

  * Stage 1: SimCLR-style InfoNCE pretrain of
    ``PerTickerTemporalEncoder`` on the fold's TRAINING days, producing
    a ``foldF_encoder.pt`` per fold.
  * Stage 2: supervised finetune of the full ``InvarSTXModel`` on that
    fold's training days (one ckpt per (fold, seed)).

This trainer replaces Stage 1 with a JOINT multi-universe contrastive
pretrain on the UNION of three universes' fold-causal training days,
sharing a single backbone (positional embedding + transformer +
LayerNorm) but routing per-universe per-stock feature widths through
PER-UNIVERSE ``Linear(F_u, d_model)`` input projections (Option (c)
from the task spec).

After Stage 1, ``assemble_per_universe_encoder_state`` rebuilds a
canonical-keyed ``PerTickerTemporalEncoder.state_dict()`` for each
universe; that state is written to a per-(universe, fold)
``foldF_encoder.pt`` file so the UNMODIFIED canonical Stage-2 loader
(``run_stage2_finetune``) can finetune on each universe independently
with a single seed.

Leakage discipline
------------------
The pretrain corpus for EACH universe is its fold's training days
ONLY (``fold_split(cfg, dates)[0]``); per-universe ``val_idx`` and
``test_idx`` are never read. Per-universe regime fingerprints are
standardised with that universe's TRAIN-day stats only. The per-fold
contrast is INSIDE each universe (positives are nearest in-batch days
in the SAME universe's standardised regime-fingerprint space); we do
NOT cross-universe-mix positives, because the regime fingerprint
distributions for SP500, NDX and NBI are NOT directly comparable
(different sectoral / volatility scaling). Cross-universe transfer
comes through the SHARED backbone trained on a balanced mix of
per-universe contrastive batches.

Per-epoch sampling
------------------
The trainer holds one ``ContrastiveCorpus`` per universe, each carrying
its own ``(x_t, valid_days, day_keys_z)`` and ``leakage_set`` for the
per-day assertion. Within each epoch we:

  1. For each universe, draw ``ceil(steps_per_epoch / N_universes)``
     non-overlapping contrastive minibatches of ``CL_BATCH_DAYS`` days
     (with replacement across epochs; no replacement within an epoch).
  2. Interleave the per-universe minibatches round-robin so the
     optimiser sees one universe's batch per step.
  3. Per-batch loss = canonical
     ``_supcon_infonce_loss`` on the universe-specific day embeddings
     (regime fingerprint POSITIVES come from the SAME universe, also
     restricted to that universe's pretrain corpus).
  4. One AdamW + ``warmup_cosine_lr`` schedule across all parameters.

Outputs
-------
For each (universe, fold) and an optional ``--save-state`` request,
writes the assembled canonical encoder state to::

    {output_dir}/_ckpt_per_universe/{universe_id}/fold{F}_encoder.pt

with the SAME payload schema used by the canonical clpretrain Stage-1
trainer so the canonical Stage-2 loader's strict load works unchanged.

CLI
---
Single-fold, 3-universe joint pretrain (one fold per invocation; the
sbatch loops over folds)::

    PYTHONPATH=$PWD python3 -m src.baselines.train_multitask_pretrain \\
        --fold 1 --seed 42 \\
        --panel_end 2025-12-31 --two_regime_val \\
        --pretrain_epochs 10 \\
        --output_dir invar_rl/results/multitask_l1
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from src.baselines.v2_runner import (
    build_masks,
    build_panel,
    fold_split,
    set_seeds,
    standardize_features,
    warmup_cosine_lr,
)
from src.baselines.train_invar_stx_v2 import InvarSTXV2Config
from src.baselines.train_invar_clpretrain_v2 import (
    CL_BATCH_DAYS,
    CL_POS_FRAC,
    CL_PROJ_DIM,
    CL_TEMPERATURE,
    _assert_pretrain_causal,
    _supcon_infonce_loss,
)
from src.v2.data.episode_keys import EpisodeKeyConfig, build_episode_keys
from src.models.multitask_l1 import (
    MultitaskTemporalEncoder,
    MultitaskTemporalEncoderConfig,
    UNIVERSE_FEATURE_DIMS,
    assemble_per_universe_encoder_state,
)


# Universes participating in the Option B joint pretrain. Keys must
# match ``UNIVERSE_FEATURE_DIMS`` AND the ``panel_kind`` strings the v2
# runner accepts; values are the panel_end strings used for that
# universe in canonical training (matches the per-universe sbatches
# under invar_rl/scripts/wulver/).
DEFAULT_UNIVERSES: List[str] = [
    "lattice_native",
    "nasdaq100",
    "biotech_nbi_enriched",
]
DEFAULT_PANEL_END: str = "2025-12-31"


@dataclass
class ContrastiveCorpus:
    """One universe's fold-causal contrastive pretrain corpus.

    Each universe owns its own ``(T, N, F_u)`` standardised panel, its
    own train/val/test fold split, its own 14-d regime fingerprint
    standardised with that universe's TRAIN-day stats, and its own
    cached ``leakage_set`` for the per-day assertion.

    Tensors stored on the trainer's compute device for fast per-step
    slicing. ``valid_days`` is the integer list of training-day indices
    that can serve as contrastive anchors (``t >= W-1`` and at least 3
    tradable tickers on day t).
    """

    universe_id: str
    cfg: InvarSTXV2Config
    x_tensor: Tensor                  # (T, N, F_u) standardised, on device
    tradable: np.ndarray              # (T, N) bool
    day_keys_z: np.ndarray            # (T, 14) standardised
    valid_days: List[int]             # training days usable as anchors
    leakage_set: set                  # set of train-day indices for assertion


def _build_corpus(
    universe_id: str,
    fold: int,
    panel_end: str,
    two_regime_val: bool,
    temporal_window: int,
    device: torch.device,
) -> ContrastiveCorpus:
    """Build one universe's fold-causal contrastive corpus.

    Mirrors ``run_stage1_pretrain``'s data prep in
    ``train_invar_clpretrain_v2.py`` (build_panel + build_masks +
    fold_split + train-fold standardisation + 14-d regime fingerprint
    with TRAIN-day stats only + leakage guard) but does NOT instantiate
    any model. The trainer owns the shared model + optimiser; each
    corpus only carries the data + metadata for one universe.
    """
    cfg = InvarSTXV2Config(fold=fold, seed=42)
    cfg.panel_kind = universe_id
    cfg.two_regime_val = bool(two_regime_val)
    cfg.panel_end = str(panel_end)
    cfg.enable_retrieval_bank = False  # bankless canonical invariant
    cfg.temporal_window = int(temporal_window)

    x_raw, _, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    expected_f = int(UNIVERSE_FEATURE_DIMS.get(universe_id, -1))
    if Fdim != expected_f:
        raise RuntimeError(
            f"[ERR] universe={universe_id} panel feature width {Fdim} "
            f"does not match UNIVERSE_FEATURE_DIMS[{universe_id}]={expected_f}; "
            f"refusing to silently broadcast."
        )
    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    pretrain_idx = np.asarray(train_idx).astype(np.int64)
    _assert_pretrain_causal(pretrain_idx, train_idx, val_idx, test_idx)
    leakage_set = set(int(i) for i in pretrain_idx.tolist())

    # Train-fold standardisation only (val/test untouched).
    x = standardize_features(x_raw, tradable, train_idx)
    x_t = torch.from_numpy(x).to(device)

    # 14-d regime fingerprint standardised with TRAIN-day stats only.
    day_keys, _ = build_episode_keys(
        dates=dates, log_returns=x_raw[..., 0], mask=tradable,
        cfg=EpisodeKeyConfig(),
    )
    key_tr = day_keys[train_idx]
    key_mu = key_tr.mean(axis=0)
    key_sd = key_tr.std(axis=0)
    key_sd = np.where(key_sd < 1e-6, 1.0, key_sd)
    day_keys_z = ((day_keys - key_mu) / key_sd).astype(np.float32)

    W = cfg.temporal_window
    valid_days = [
        int(t) for t in pretrain_idx
        if int(t) >= W - 1 and tradable[int(t)].sum() >= 3
    ]

    print(
        f"[multitask S1] {universe_id} fold={fold}: T={T} N={N} F={Fdim} "
        f"train_days={len(train_idx)} valid_anchors={len(valid_days)}",
        flush=True,
    )
    return ContrastiveCorpus(
        universe_id=universe_id,
        cfg=cfg,
        x_tensor=x_t,
        tradable=tradable,
        day_keys_z=day_keys_z,
        valid_days=valid_days,
        leakage_set=leakage_set,
    )


def _draw_universe_batch(
    corpus: ContrastiveCorpus,
    rng: np.random.RandomState,
    batch_days: int,
) -> np.ndarray:
    """Sample ``batch_days`` non-overlapping training days for one
    universe's contrastive minibatch. Returns int64 day indices.

    No-replacement WITHIN a draw; epoch-level reshuffling is the
    trainer's responsibility (it draws multiple batches per epoch).
    """
    if len(corpus.valid_days) < batch_days:
        raise RuntimeError(
            f"[ERR] universe={corpus.universe_id}: only "
            f"{len(corpus.valid_days)} valid anchor days but batch="
            f"{batch_days} requested."
        )
    perm = rng.permutation(np.asarray(corpus.valid_days, dtype=np.int64))
    return perm[:batch_days]


def _compute_pos_mask(
    day_keys_z: np.ndarray,
    batch: np.ndarray,
    n_pos: int,
    device: torch.device,
) -> Tensor:
    """Build the supervised-contrastive positive mask for one batch.

    Mirrors the per-day positive selection in
    ``train_invar_clpretrain_v2.run_stage1_pretrain``: nearest in-batch
    days by L2 in standardised regime-fingerprint space (self excluded).
    """
    keys_b = torch.from_numpy(day_keys_z[batch]).float().to(device)
    with torch.no_grad():
        kd = torch.cdist(keys_b, keys_b)
        bb = kd.shape[0]
        eye = torch.eye(bb, dtype=torch.bool, device=device)
        kd = kd.masked_fill(eye, float("inf"))
        k = min(n_pos, bb - 1)
        nn_idx = torch.topk(kd, k=k, dim=1, largest=False).indices
        pos_mask = torch.zeros(bb, bb, dtype=torch.bool, device=device)
        pos_mask.scatter_(1, nn_idx, True)
        pos_mask = pos_mask & (~eye)
    return pos_mask


def _per_universe_batch_loss(
    encoder: MultitaskTemporalEncoder,
    proj_head: nn.Module,
    corpus: ContrastiveCorpus,
    batch: np.ndarray,
    n_pos: int,
    tau: float,
    device: torch.device,
) -> Tensor:
    """Forward + InfoNCE loss for one universe's contrastive minibatch.

    Per-day assertion: every index in ``batch`` MUST lie in this
    universe's pretrain corpus (train_idx). The assertion mirrors the
    canonical clpretrain trainer's leakage guard.
    """
    # Per-day leakage assertion (same as canonical clpretrain).
    for _t in batch:
        if int(_t) not in corpus.leakage_set:
            raise RuntimeError(
                f"[ERR] LEAKAGE: universe={corpus.universe_id} S1 used "
                f"day {int(_t)} not in train_idx."
            )
    pos_mask = _compute_pos_mask(corpus.day_keys_z, batch, n_pos, device)
    W = corpus.cfg.temporal_window
    z_list: List[Tensor] = []
    for _t in batch:
        t = int(_t)
        m_np = corpus.tradable[t]
        active_idx = np.flatnonzero(m_np)
        active_t = torch.from_numpy(active_idx).to(device)
        # (N_active, T, F_u) lookback window.
        x_win = corpus.x_tensor[t - W + 1: t + 1, active_t, :].transpose(0, 1)
        z = encoder.day_embedding(x_win, corpus.universe_id, proj_head)
        z_list.append(z)
    z = torch.stack(z_list, dim=0)  # (B, proj_dim)
    return _supcon_infonce_loss(z, pos_mask, tau)


def run_multitask_pretrain(
    universes: Sequence[str],
    fold: int,
    seed: int,
    pretrain_epochs: int,
    output_dir: Path,
    panel_end: str,
    two_regime_val: bool,
    learning_rate: float = 1.0e-4,
    weight_decay: float = 1.0e-5,
    warmup_steps: int = 500,
    grad_clip: float = 1.0,
    temporal_window: int = 20,
    d_model: int = 128,
    n_heads: int = 4,
    d_ff: int = 256,
    e_layers: int = 2,
    dropout: float = 0.1,
    activation: str = "gelu",
    device: torch.device | None = None,
) -> Dict[str, Path]:
    """Run Option B Stage-1 multi-universe contrastive pretrain.

    Returns the per-universe encoder ckpt paths so the per-universe
    finetune driver can pick them up by id. Each ckpt is written to::

        {output_dir}/_ckpt_per_universe/{universe_id}/fold{F}_encoder.pt

    in the SAME schema the canonical clpretrain Stage-1 trainer uses,
    so the canonical Stage-2 loader works unchanged.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(seed)

    if not universes:
        raise ValueError("[ERR] at least one universe required")
    for u in universes:
        if u not in UNIVERSE_FEATURE_DIMS:
            raise ValueError(
                f"[ERR] universe={u!r} not registered in "
                f"UNIVERSE_FEATURE_DIMS={sorted(UNIVERSE_FEATURE_DIMS)}"
            )

    # ---- Build per-universe corpora. ----
    corpora: Dict[str, ContrastiveCorpus] = {}
    for uid in universes:
        corpora[uid] = _build_corpus(
            universe_id=uid,
            fold=fold,
            panel_end=panel_end,
            two_regime_val=two_regime_val,
            temporal_window=temporal_window,
            device=device,
        )

    # ---- Build shared encoder + projection head. ----
    enc_cfg = MultitaskTemporalEncoderConfig(
        temporal_window=temporal_window,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        e_layers=e_layers,
        dropout=dropout,
        activation=activation,
        universe_feature_dims={
            u: UNIVERSE_FEATURE_DIMS[u] for u in universes
        },
    )
    encoder = MultitaskTemporalEncoder(enc_cfg).to(device)
    # SimCLR 2-layer projection head (d -> d -> proj_dim); shared across
    # universes by design so the contrastive feature space is common.
    proj_head = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, CL_PROJ_DIM),
    ).to(device)

    # ---- Optimiser + scheduler covering BOTH encoder and proj head. ----
    params = list(encoder.parameters()) + list(proj_head.parameters())
    optim = torch.optim.AdamW(
        params, lr=learning_rate, weight_decay=weight_decay,
    )
    # Per epoch: each universe contributes
    # floor(len(valid_days) / CL_BATCH_DAYS) minibatches; total steps
    # per epoch = sum across universes; total_steps for the schedule.
    per_uni_batches = {
        u: max(1, len(corpora[u].valid_days) // CL_BATCH_DAYS)
        for u in universes
    }
    steps_per_epoch = sum(per_uni_batches.values())
    total_steps = pretrain_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda s: warmup_cosine_lr(s, warmup_steps, total_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    n_pos = max(1, int(np.ceil(CL_POS_FRAC * CL_BATCH_DAYS)))
    print(
        f"[multitask S1] fold={fold} seed={seed} universes={list(universes)} "
        f"epochs={pretrain_epochs} steps/epoch={steps_per_epoch} "
        f"per_uni_batches={per_uni_batches} batch={CL_BATCH_DAYS} "
        f"pos/anchor={n_pos} tau={CL_TEMPERATURE}",
        flush=True,
    )

    # ---- Joint training loop. ----
    encoder.train()
    proj_head.train()
    for epoch in range(pretrain_epochs):
        t0 = time.time()
        rng = np.random.RandomState(seed + epoch)
        # Pre-draw all per-universe minibatches for this epoch (no
        # replacement within universe), then round-robin them so the
        # optimiser alternates universes step-by-step.
        per_uni_batches_idx: Dict[str, List[np.ndarray]] = {}
        for uid in universes:
            corp = corpora[uid]
            local_rng = np.random.RandomState(rng.randint(2 ** 30))
            perm = local_rng.permutation(
                np.asarray(corp.valid_days, dtype=np.int64)
            )
            batches: List[np.ndarray] = []
            B = CL_BATCH_DAYS
            for b0 in range(0, len(perm) - B + 1, B):
                batches.append(perm[b0: b0 + B])
            per_uni_batches_idx[uid] = batches

        # Round-robin schedule: at step s, pick universe s % U if it has
        # any remaining batch; skip otherwise.
        cursors = {u: 0 for u in universes}
        losses_per_uni: Dict[str, List[float]] = {u: [] for u in universes}
        step_order: List[str] = []
        remaining = sum(len(b) for b in per_uni_batches_idx.values())
        u_cycle = list(universes)
        ci = 0
        while remaining > 0:
            uid = u_cycle[ci % len(u_cycle)]
            ci += 1
            if cursors[uid] < len(per_uni_batches_idx[uid]):
                step_order.append(uid)
                cursors[uid] += 1
                remaining -= 1

        # Replay with a fresh cursor map.
        cursors = {u: 0 for u in universes}
        for uid in step_order:
            batch = per_uni_batches_idx[uid][cursors[uid]]
            cursors[uid] += 1
            with torch.amp.autocast(
                "cuda", enabled=(device.type == "cuda")
            ):
                loss = _per_universe_batch_loss(
                    encoder=encoder,
                    proj_head=proj_head,
                    corpus=corpora[uid],
                    batch=batch,
                    n_pos=n_pos,
                    tau=CL_TEMPERATURE,
                    device=device,
                )
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            losses_per_uni[uid].append(float(loss.item()))

        dt = time.time() - t0
        msg_parts = [f"epoch {epoch}"]
        for uid in universes:
            ls = losses_per_uni[uid]
            mean_l = float(np.mean(ls)) if ls else float("nan")
            msg_parts.append(f"{uid}={mean_l:.5f}({len(ls)})")
        msg_parts.append(f"({dt:.1f}s)")
        print("[multitask S1] " + " ".join(msg_parts), flush=True)

    # ---- Save per-universe canonical encoder ckpts. ----
    out_root = Path(output_dir) / "_ckpt_per_universe"
    out_paths: Dict[str, Path] = {}
    for uid in universes:
        uni_dir = out_root / uid
        uni_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = uni_dir / f"fold{fold}_encoder.pt"
        enc_state = assemble_per_universe_encoder_state(encoder, uid)
        torch.save(
            {
                "fold": int(fold),
                "seed": int(seed),
                "pretrain_epochs": int(pretrain_epochs),
                "panel_kind": uid,
                "encoder_state_dict": enc_state,
                "multitask_universes": list(universes),
                "multitask_pretrain": True,
            },
            ckpt_path,
        )
        out_paths[uid] = ckpt_path
        print(
            f"[multitask S1] saved {uid} encoder ckpt -> {ckpt_path}",
            flush=True,
        )
    return out_paths


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Option B Stage-1 multi-universe contrastive pretrain of "
            "the canonical PerTickerTemporalEncoder backbone."
        )
    )
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--universes",
        type=str,
        default=",".join(DEFAULT_UNIVERSES),
        help=(
            "Comma-separated panel_kind ids participating in the "
            "joint pretrain. Default = SP500 + NDX + NBI."
        ),
    )
    p.add_argument(
        "--panel_end",
        type=str,
        default=DEFAULT_PANEL_END,
        help="Shared panel_end for all universes (canonical = 2025-12-31).",
    )
    p.add_argument(
        "--two_regime_val", action="store_true", default=True,
        help=(
            "Use the two_regime_val fold split for every universe "
            "(val = 2017H2 + 2018H2). Default ON per canonical."
        ),
    )
    p.add_argument(
        "--pretrain_epochs", type=int, default=10,
        help="Stage-1 epochs (default 10 = canonical).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="invar_rl/results/multitask_l1",
        help=(
            "Per-universe ckpts written to "
            "{output_dir}/_ckpt_per_universe/{universe}/fold{F}_encoder.pt."
        ),
    )
    p.add_argument(
        "--learning_rate", type=float, default=1.0e-4,
    )
    p.add_argument("--weight_decay", type=float, default=1.0e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    universes = [u.strip() for u in args.universes.split(",") if u.strip()]
    print(
        f"[INFO] multitask pretrain fold={args.fold} seed={args.seed} "
        f"universes={universes} epochs={args.pretrain_epochs} "
        f"device={device}",
        flush=True,
    )
    run_multitask_pretrain(
        universes=universes,
        fold=args.fold,
        seed=args.seed,
        pretrain_epochs=args.pretrain_epochs,
        output_dir=Path(args.output_dir),
        panel_end=args.panel_end,
        two_regime_val=bool(args.two_regime_val),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_steps=int(args.warmup_steps),
        grad_clip=float(args.grad_clip),
        device=device,
    )


if __name__ == "__main__":
    main()
