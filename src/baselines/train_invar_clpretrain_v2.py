"""Two-stage fold-causal CONTRASTIVE pretrain -> finetune for the
canonical BANKLESS InVAR (EXPERIMENT-ONLY; never in any paper).

NOT a paper baseline. This trainer is a COPY of
``src.baselines.train_invar_pretrain_v2`` with EXACTLY ONE change: the
Stage-1 self-supervised objective is REGIME-CONTRASTIVE instead of
masked-window reconstruction. Everything else (the fold-causal corpus
restriction ``pretrain_idx = fold_split(cfg, dates)[0]``, the
``_assert_pretrain_causal`` hard leakage guard, train-fold
standardisation, the Stage-2 finetune that loads the fold encoder ckpt
into the canonical BANKLESS InVAR with a layer-wise LR + two-regime val
+ SWA + cs_mse_loss, the JSON-only disk-safe write, the argparse, the
bankless asserts, the encoder-ckpt path/key convention) is reused
BYTE-IDENTICAL so the leakage discipline is preserved and the new bug
surface is the Stage-1 loss only.

Design
------
Stage 1 (REGIME-CONTRASTIVE SSL of the per-ticker temporal encoder):
    A small pretrain wrapper holds the SAME ``PerTickerTemporalEncoder``
    submodules as the canonical model plus a SimCLR-style 2-layer MLP
    projection head (``d -> d -> proj_dim``). For each training day t
    (restricted to ``pretrain_idx`` ONLY) the encoder runs over that
    day's active-ticker ``(N, T, F)`` windows; the per-ticker ``(N, d)``
    last-step outputs are masked-mean-pooled (active tickers only) into
    a single day embedding, projected through the head, and
    L2-normalised -> ``z_t``.

    POSITIVES: each training day carries the SAME 14-d regime
    fingerprint that the day-memory uses (8-dim risk + 6-dim
    cross-sectional diagnostics from ``build_episode_keys``, built
    causally from the panel and restricted to ``pretrain_idx`` training
    days; standardised with TRAIN-day stats only). Within a minibatch of
    training days the positives of an anchor day are the ``P`` other
    in-batch days whose standardised regime key is nearest (L2) to the
    anchor's, ``P = ceil(pos_frac * batch)``. NEGATIVES are the other
    in-batch days. The loss is a supervised-contrastive / InfoNCE
    objective with cosine similarity and temperature ``tau``:
        L = -mean_i log( sum_{p in pos(i)} exp(sim(i,p)/tau)
                          / sum_{a != i} exp(sim(i,a)/tau) ).
    The temporal encoder + projection head are trained with AdamW +
    ``warmup_cosine_lr`` (fp16 ok). ``cfg.pretrain_epochs`` (default 10).

    LEAKAGE: the pretrain corpus is EXACTLY the fold's training days
    ``train_idx = fold_split(cfg, dates)[0]``; ``val_idx`` /
    ``test_idx`` are never read. The regime fingerprint is standardised
    with TRAIN-day stats only and the (optional) k-means clustering is
    fit on TRAIN days only. ``_assert_pretrain_causal`` enforces the
    subset/disjoint invariant, and every day index used inside Stage-1
    is asserted to lie in ``pretrain_idx``.

    Only the temporal-encoder ``state_dict`` is saved (the projection
    head is discarded) to the SAME fold-keyed checkpoint
    ``results/<out>/_ckpt/foldF_encoder.pt`` with the SAME key
    convention as the masked-recon trainer, so Stage-2's strict load is
    unchanged.

Stage 2 (finetune the full BANKLESS InVAR per (fold, seed)):
    BYTE-IDENTICAL to ``train_invar_pretrain_v2.run_stage2_finetune``
    (same v2_runner calls, same SWA EMA loop, same early-stop, same JSON
    schema, same strict ckpt load, same layer-wise LR).

Disk: experiment-only. Stage 2 writes ONLY the
``fold{F}_seed{S}.json`` (history entries contain ``"epoch"`` so the
sbatch skip-if-done test passes). NO predictions npz. Stage 1 writes
ONLY the small encoder checkpoint .pt.

Run (1-fold smoke):
    # Stage 1: contrastive-pretrain fold 1 (1 epoch), save ckpt, exit.
    python -m src.baselines.train_invar_clpretrain_v2 --fold 1 --seed 42 \
        --panel_kind lattice_native --two_regime_val \
        --pretrain_only --pretrain_epochs 1 \
        --output_dir results/invar_clpretrain
    # Stage 2: finetune fold 1 seed 42 (1 epoch) from the checkpoint.
    python -m src.baselines.train_invar_clpretrain_v2 --fold 1 --seed 42 \
        --panel_kind lattice_native --two_regime_val \
        --skip_pretrain --finetune_epochs 1 \
        --output_dir results/invar_clpretrain
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from src.baselines.v2_runner import (
    build_age_features,
    build_masks,
    build_panel,
    cs_mse_loss,
    evaluate_predictions,
    fold_split,
    set_seeds,
    standardize_features,
    warmup_cosine_lr,
)
from src.baselines.train_invar_stx_v2 import (
    InvarSTXModel,
    InvarSTXV2Config,
    PerTickerTemporalEncoder,
)
# NOTE: ``src.invar.training.loss.listmle_loss`` is imported lazily inside
# ``run_stage2_finetune`` to break the circular import between this module
# and ``src.invar.canonical`` (which re-exports the contrastive pretrainer
# defined below). Importing it at module-load time triggers
# ``src.invar.__init__.py`` which imports back into this module before its
# class definitions are complete.
from src.v2.data.episode_keys import (
    EPISODE_KEY_COLS,
    EpisodeKeyConfig,
    build_episode_keys,
)
from src.v2.data.macro_duration_features import (
    MACRO_GATE_COLS,
    build_macro_duration_features,
    standardize_macro_duration,
)
from src.v2.data.rolling_macro_betas import (
    betas_to_tensor,
    build_rolling_betas,
)
from src.v2.training.train_dow_epistar import (
    resolve_duration_indices,
    _gather_or_zero,
)


# ============================================================================
# STAGE 1: REGIME-CONTRASTIVE self-supervised pretrain of the per-ticker
# temporal encoder.
# ============================================================================

# Contrastive-pretrain hyperparameters (Stage-1 only; the masked-recon
# trainer used a 0.5 mask ratio here instead). InfoNCE temperature, the
# SimCLR projection-head width, and the fraction of the in-batch days
# treated as positives for each anchor (nearest in standardised
# regime-fingerprint space).
CL_PROJ_DIM = 128
CL_TEMPERATURE = 0.1
CL_POS_FRAC = 0.1
CL_BATCH_DAYS = 64


class TemporalEncoderContrastivePretrainer(nn.Module):
    """Regime-contrastive (SimCLR / SupCon-style) wrapper around the
    canonical ``PerTickerTemporalEncoder``.

    Holds its OWN ``PerTickerTemporalEncoder`` instance (built with the
    SAME constructor args the finetune model will use, so its
    ``state_dict`` keys match ``model.temporal_encoder`` for the Stage-2
    strict load) plus a 2-layer MLP projection head
    (``d -> d -> proj_dim``). The pretext produces ONE projected,
    L2-normalised embedding per training day from that day's
    active-ticker windows; the contrastive loss lives outside this
    module. After pretrain, ``encoder.state_dict()`` is the artefact
    loaded into the finetune model's ``temporal_encoder`` submodule
    (strict key match); the projection head is discarded.

    B2 (2026-05-27): when ``aux_regression_head`` is True a small
    ``nn.Linear(d_model, 1)`` per-ticker regression head is also
    instantiated. It produces a scalar score per active ticker from
    the SAME ``per_ticker`` encoder output that ``day_embedding`` uses.
    The head is discarded after Stage 1; only the encoder is carried
    into Stage 2.
    """

    def __init__(
        self,
        n_features: int,
        temporal_window: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        dropout: float,
        activation: str,
        proj_dim: int,
        aux_regression_head: bool = False,
        masked_feature_head: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = PerTickerTemporalEncoder(
            n_features=n_features,
            temporal_window=temporal_window,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            dropout=dropout,
            activation=activation,
        )
        # SimCLR-style 2-layer projection head (d -> d -> proj_dim).
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
        # B2: optional auxiliary regression head on the per-ticker
        # encoder output (d_model -> 1). Only created when requested
        # so the canonical pretrainer state_dict / parameter list is
        # unchanged when the flag is off.
        self.aux_regression_head_enabled = bool(aux_regression_head)
        if self.aux_regression_head_enabled:
            self.aux_reg_head = nn.Linear(d_model, 1)
        else:
            self.aux_reg_head = None
        # C2 (2026-05-27): optional masked-feature reconstruction head
        # (d_model -> n_features). Only created when requested so the
        # canonical pretrainer state_dict / parameter list is unchanged
        # when the flag is off. The head is discarded after Stage 1;
        # only the encoder is loaded into Stage 2.
        self.masked_feature_head_enabled = bool(masked_feature_head)
        if self.masked_feature_head_enabled:
            from src.models.pretrain_improvements.masked_feature_modeling \
                import MaskedFeatureHead
            self.masked_feature_head = MaskedFeatureHead(
                d_model=d_model, feature_dim=n_features,
            )
        else:
            self.masked_feature_head = None
        # Track the encoder's input feature dim so the C2 trainer can
        # build masks of the right width without reaching into the
        # encoder internals.
        self._n_features = int(n_features)

    def day_embedding(self, x_window: Tensor) -> Tensor:
        """Encode one day's active-ticker windows into ONE day vector.

        Args:
            x_window: ``(N_active, T, F)`` per-ticker lookback windows
                for the active tickers of a single training day.

        Returns:
            ``(proj_dim,)`` L2-normalised projected day embedding.
        """
        # Per-ticker last-step encodings via the canonical encoder
        # forward (returns (N_active, d_model); the active mask was
        # already applied by selecting active tickers upstream, so a
        # plain mean over the ticker axis is the masked mean-pool).
        per_ticker = self.encoder(x_window)                    # (N, d)
        day_vec = per_ticker.mean(dim=0)                       # (d,)
        z = self.proj_head(day_vec)                            # (proj_dim,)
        z = torch.nn.functional.normalize(z, dim=-1)
        return z

    def per_ticker_projections(self, x_window: Tensor) -> Tensor:
        """C3 (2026-05-27): per-stock projected embeddings (no day pool).

        Args:
            x_window: ``(N_active, T, F)`` per-ticker lookback windows
                for the active tickers of a single training day.

        Returns:
            ``(N_active, proj_dim)`` L2-normalised projected per-stock
            embeddings produced by the SAME encoder + SAME projection
            head used by :meth:`day_embedding`. No pooling.
        """
        per_ticker = self.encoder(x_window)                    # (N, d)
        z = self.proj_head(per_ticker)                         # (N, proj)
        z = torch.nn.functional.normalize(z, dim=-1)
        return z

    def day_and_stock_projections(
        self, x_window: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """A4 (2026-05-27): both the day-level and per-stock projections
        from a SINGLE encoder forward pass.

        The joint multi-objective Stage-1 pretrain needs the day-level
        regime InfoNCE term (which consumes the pooled day embedding) and
        the per-stock SupCon terms (sector + co-movement) in the same
        batch. Computing both from one ``self.encoder(x_window)`` call
        avoids a redundant second forward and keeps the day vector and
        the per-stock vectors consistent.

        Args:
            x_window: ``(N_active, T, F)`` per-ticker lookback windows.

        Returns:
            ``(z_day, z_stocks)`` where ``z_day`` is the ``(proj_dim,)``
            L2-normalised pooled day embedding (same definition as
            :meth:`day_embedding`) and ``z_stocks`` is the
            ``(N_active, proj_dim)`` L2-normalised per-stock projections
            (same definition as :meth:`per_ticker_projections`).
        """
        per_ticker = self.encoder(x_window)                    # (N, d)
        day_vec = per_ticker.mean(dim=0)                       # (d,)
        z_day = torch.nn.functional.normalize(
            self.proj_head(day_vec), dim=-1,
        )                                                      # (proj,)
        z_stocks = torch.nn.functional.normalize(
            self.proj_head(per_ticker), dim=-1,
        )                                                      # (N, proj)
        return z_day, z_stocks

    def day_embedding_with_scores(
        self, x_window: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Same as :meth:`day_embedding` but ALSO returns per-ticker
        scalar scores from the B2 auxiliary regression head.

        Requires ``aux_regression_head_enabled`` to be True; raises
        otherwise to keep the canonical (flag-off) path call-site free
        of accidental head creation.

        Args:
            x_window: ``(N_active, T, F)`` per-ticker lookback windows.

        Returns:
            ``(z, scores)`` where ``z`` is the ``(proj_dim,)``
            L2-normalised projected day embedding and ``scores`` is the
            ``(N_active,)`` per-ticker scalar score from the aux head.
        """
        if not self.aux_regression_head_enabled:
            raise RuntimeError(
                "day_embedding_with_scores requires aux_regression_head"
                "_enabled=True; canonical pretrain path has no aux head."
            )
        per_ticker = self.encoder(x_window)                    # (N, d)
        day_vec = per_ticker.mean(dim=0)                       # (d,)
        z = self.proj_head(day_vec)                            # (proj_dim,)
        z = torch.nn.functional.normalize(z, dim=-1)
        scores = self.aux_reg_head(per_ticker).squeeze(-1)     # (N,)
        return z, scores

    def reconstruct_masked_features(
        self, x_window_masked: Tensor,
    ) -> Tensor:
        """C2 (2026-05-27): encode a masked window and reconstruct features.

        Args:
            x_window_masked: ``(N_active, T, F)`` lookback window with
                masked positions zeroed on the last time step.

        Returns:
            ``(N_active, F)`` reconstructed feature row at the LAST
            time step of each stock's lookback.
        """
        if not self.masked_feature_head_enabled:
            raise RuntimeError(
                "reconstruct_masked_features requires masked_feature_"
                "head=True; canonical pretrain path has no MFM head."
            )
        per_ticker = self.encoder(x_window_masked)             # (N, d)
        return self.masked_feature_head(per_ticker)            # (N, F)


def _supcon_infonce_loss(
    z: Tensor,
    pos_mask: Tensor,
    tau: float,
) -> Tensor:
    """Supervised-contrastive / InfoNCE loss over a minibatch of days.

    Args:
        z: ``(B, proj_dim)`` L2-normalised day embeddings.
        pos_mask: ``(B, B)`` bool; ``pos_mask[i, j]`` True iff day j is a
            positive for anchor i (always False on the diagonal).
        tau: softmax temperature.

    Returns:
        Scalar loss
        ``-mean_i log( sum_{p in pos(i)} exp(sim(i,p)/tau)
                       / sum_{a != i} exp(sim(i,a)/tau) )``
        with ``sim`` = cosine similarity (z is already L2-normalised so
        ``z @ z.T`` is cosine). Anchors with no positive are skipped.
    """
    b = z.shape[0]
    sim = (z @ z.t()) / max(tau, 1e-6)                         # (B, B)
    self_mask = torch.eye(b, dtype=torch.bool, device=z.device)
    # Numerically-stable log-sum-exp over all non-self entries.
    sim_masked = sim.masked_fill(self_mask, float("-inf"))
    logits_max = sim_masked.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim_masked - logits_max)
    denom = exp_sim.sum(dim=1)                                 # (B,)
    log_prob = (sim - logits_max.squeeze(1).unsqueeze(1)
                - torch.log(denom.clamp_min(1e-12)).unsqueeze(1))
    pos = pos_mask & (~self_mask)
    pos_counts = pos.sum(dim=1)                                # (B,)
    valid = pos_counts > 0
    if not bool(valid.any()):
        return torch.zeros((), device=z.device, dtype=z.dtype)
    # Mean log-likelihood of the positives per anchor, averaged over
    # anchors that actually have at least one positive.
    pos_log_prob = (log_prob * pos.float()).sum(dim=1)
    per_anchor = pos_log_prob[valid] / pos_counts[valid].clamp_min(1).float()
    return -per_anchor.mean()


def _supcon_infonce_loss_per_day(
    z_per_stock: Tensor,
    pos_mask: Tensor,
    tau: float,
) -> Tensor:
    """C3 (2026-05-27): per-day SupCon InfoNCE across stocks.

    Same numerical formula as :func:`_supcon_infonce_loss` but the
    contrastive cohort is the (N_active,) active cross-section of ONE
    trading day; positives = same-sector peers, negatives = different-
    sector peers, both inside the same day. Anchors with no positive
    are skipped (matches the canonical no-positive handling).

    Args:
        z_per_stock: ``(N_active, proj_dim)`` L2-normalised per-stock
            projected embeddings for ONE day.
        pos_mask: ``(N_active, N_active)`` bool; ``pos_mask[i, j]``
            True iff stock j is a positive for anchor i (always False
            on the diagonal). Unknown-sector anchors get all-False rows.
        tau: softmax temperature.

    Returns:
        Scalar loss, mean over anchors with >=1 positive. If no anchor
        has any positive, returns a 0.0 tensor on the same device/dtype.
    """
    return _supcon_infonce_loss(z_per_stock, pos_mask, tau)


def run_stage1_pretrain(
    cfg: InvarSTXV2Config,
    pretrain_epochs: int,
    device: torch.device,
    ckpt_path: Path,
    init_from_ckpt: bool = False,
) -> None:
    """Fold-causal REGIME-CONTRASTIVE self-supervised pretrain of the
    temporal encoder.

    The pretrain corpus is restricted to ``train_idx`` ONLY (the fold's
    training days). ``val_idx`` / ``test_idx`` are never read; the
    regime fingerprint is standardised with TRAIN-day stats only; an
    explicit leakage assertion plus a per-day in-corpus assertion
    enforce this.

    ``init_from_ckpt`` (A1 2026-05-27, multi-stage sequential pretrain):
    when True, the temporal encoder is initialised from the weights
    already saved at ``ckpt_path`` instead of fresh random init. This is
    how Stage 1b of an ["regime", "sector"] curriculum continues
    training the Stage 1a encoder under a different positive selector.
    Default False preserves the canonical single-stage path byte-
    identically.
    """
    set_seeds(cfg.seed)

    # ---- v2_runner data / fold calls: SAME args, SAME order as
    # train_invar_pretrain_v2.py / train_invar_stx_v2.py. ----
    x_raw, y, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    print(f"[InVAR-clpretrain S1] panel: T={T} N={N} F={Fdim}")
    min_n = 25 if cfg.panel_kind == "djia30" else 50
    if N < min_n:
        raise RuntimeError(
            f"Panel too small (N={N}, expected >={min_n} for "
            f"panel_kind={cfg.panel_kind})"
        )

    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    loss_mask = mm["loss_mask"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    print(f"[InVAR-clpretrain S1] fold {cfg.fold}: "
          f"train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)}")

    # ---- LEAKAGE GUARD. The pretrain corpus is EXACTLY train_idx. ----
    pretrain_idx = np.asarray(train_idx).astype(np.int64)   # corpus == train_idx
    _assert_pretrain_causal(pretrain_idx, train_idx, val_idx, test_idx)
    pretrain_set = set(int(i) for i in pretrain_idx.tolist())

    # Train-fold standardisation stats only (val/test never used here).
    x = standardize_features(x_raw, tradable, train_idx)
    x_t = torch.from_numpy(x).to(device)

    # ---- 14-d regime fingerprint: the SAME key the day-memory uses
    # (8 risk + 6 cross-sectional diagnostics). Built causally from the
    # panel exactly as train_invar_pretrain_v2 / episode_keys do, then
    # STANDARDISED WITH TRAIN-DAY STATS ONLY (val/test never touched).
    day_keys, _ = build_episode_keys(
        dates=dates, log_returns=x_raw[..., 0], mask=tradable,
        cfg=EpisodeKeyConfig(),
    )
    key_tr = day_keys[train_idx]
    key_mu = key_tr.mean(axis=0)
    key_sd = key_tr.std(axis=0)
    key_sd = np.where(key_sd < 1e-6, 1.0, key_sd)
    day_keys_z = ((day_keys - key_mu) / key_sd).astype(np.float32)
    print(f"[InVAR-clpretrain S1] regime fingerprint: "
          f"{day_keys_z.shape[1]} dims ({len(EPISODE_KEY_COLS)} cols), "
          f"train-day standardised")

    # ---- B1 (2026-05-27): optional HMM posterior selector. When the
    # config flag pretrain_regime_method == "kmeans" (default) this
    # block is a NO-OP and the canonical L2-nearest-neighbour selector
    # over day_keys_z is used below; the rest of Stage 1 is byte-
    # identical to the canonical path. When == "hmm" we fit a Gaussian
    # HMM on TRAIN-day day_keys_z only and emit per-day posteriors over
    # n_states latent regimes; the InfoNCE positive mask is then built
    # by cosine similarity over posteriors, thresholded at
    # cfg.pretrain_hmm_positive_threshold.
    pretrain_regime_method = str(
        getattr(cfg, "pretrain_regime_method", "kmeans")
    ).lower()
    if pretrain_regime_method not in ("kmeans", "hmm"):
        raise ValueError(
            "cfg.pretrain_regime_method must be 'kmeans' or 'hmm'; "
            f"got {pretrain_regime_method!r}"
        )
    day_posteriors_train_only: np.ndarray | None = None
    hmm_universe_id: str | None = None
    hmm_positive_threshold: float = 0.7
    if pretrain_regime_method == "hmm":
        from src.models.pretrain_improvements.hmm_regime import (
            HMMRegimeConfig,
            HMMRegimeLabeler,
            save_posteriors,
        )
        n_states = int(getattr(cfg, "pretrain_hmm_n_states", 4))
        hmm_positive_threshold = float(
            getattr(cfg, "pretrain_hmm_positive_threshold", 0.7)
        )
        hmm_universe_id = str(
            getattr(cfg, "pretrain_hmm_universe_id", "sp500")
        )
        hmm_cfg = HMMRegimeConfig(
            n_states=n_states,
            positive_threshold=hmm_positive_threshold,
            seed=int(cfg.seed),
        )
        labeler = HMMRegimeLabeler(hmm_cfg)
        # LEAKAGE: fit on TRAIN-day fingerprints ONLY.
        train_keys_z = day_keys_z[train_idx].astype(np.float64)
        labeler.fit(train_keys_z, n_states=n_states)
        # Posteriors for ALL training days (val/test never queried in
        # Stage 1). Cache them so the audit can read them back.
        day_posteriors_train_only = labeler.predict_proba(
            train_keys_z
        ).astype(np.float64)
        # Sanity: rows sum to 1, no NaN.
        row_sums = day_posteriors_train_only.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1.0e-6), (
            "HMM posteriors do not sum to 1; check fit data."
        )
        assert np.all(np.isfinite(day_posteriors_train_only)), (
            "HMM posteriors contain NaN/Inf."
        )
        save_posteriors(
            day_indices=np.asarray(train_idx, dtype=np.int64),
            posteriors=day_posteriors_train_only,
            universe=hmm_universe_id,
            fold=int(cfg.fold),
            backend=str(labeler.backend),
            n_states=n_states,
            n_train_days=int(labeler.n_train_days),
            seed=int(cfg.seed),
        )
        print(
            f"[InVAR-clpretrain S1] B1 HMM regime selector ON: "
            f"backend={labeler.backend} n_states={n_states} "
            f"threshold={hmm_positive_threshold:.2f} "
            f"converged={labeler.converged} "
            f"posteriors cached for universe={hmm_universe_id} "
            f"fold={cfg.fold}"
        )

    # ---- C3 (2026-05-27): optional SECTOR-AWARE per-stock InfoNCE
    # selector. When pretrain_positive_method == "regime" (default) the
    # block below is a NO-OP and the canonical / B1 day-level selector
    # path is used. When == "sector" we load the universe sector map
    # once, map this fold's panel tickers to sector ids, and the batch
    # loop runs a per-day per-stock InfoNCE (positives = same-day same-
    # sector peers, negatives = same-day different-sector peers).
    # A2 (2026-05-27): adds == "comovement"; data-driven universe-
    # agnostic version of C3 in which the per-stock cohort id comes
    # from a per-fold 252-day rolling correlation k-means cluster id
    # (see src/models/pretrain_improvements/comovement_clustering.py).
    # A4 (2026-05-27): adds == "joint"; a single Stage-1 objective that
    # aggregates the day-level regime InfoNCE term AND BOTH per-stock
    # SupCon terms (sector + co-movement) in every batch. Joint loads
    # both cohort tensors up front (sector ids into ticker_sector_ids_t,
    # co-movement cluster ids into ticker_comove_ids_t).
    pretrain_positive_method = str(
        getattr(cfg, "pretrain_positive_method", "regime")
    ).lower()
    if pretrain_positive_method not in (
        "regime", "sector", "comovement", "joint",
    ):
        raise ValueError(
            "cfg.pretrain_positive_method must be 'regime', 'sector', "
            f"'comovement', or 'joint'; got {pretrain_positive_method!r}"
        )
    is_joint = pretrain_positive_method == "joint"
    ticker_sector_ids_t: Tensor | None = None
    ticker_comove_ids_t: Tensor | None = None
    sector_universe_id: str | None = None
    if pretrain_positive_method == "sector" or is_joint:
        from src.models.pretrain_improvements.sector_positives import (
            UNKNOWN_SECTOR_ID,
            coverage_fraction,
            map_tickers_to_sector_ids,
        )
        sector_universe_id = str(
            getattr(cfg, "pretrain_sector_universe_id", "sp500")
        )
        ticker_sector_ids = map_tickers_to_sector_ids(
            list(tickers), universe=sector_universe_id,
        )
        cov = coverage_fraction(list(tickers), universe=sector_universe_id)
        if cov < 0.95:
            raise RuntimeError(
                f"C3 sector coverage too low: {cov*100:.2f}% < 95%. "
                f"Extend GICS_SECTOR_SUPPLEMENT in src/models/pretrain_"
                f"improvements/sector_positives.py."
            )
        ticker_sector_ids_t = torch.from_numpy(
            ticker_sector_ids.astype(np.int64)
        ).to(device)
        n_unknown = int((ticker_sector_ids == UNKNOWN_SECTOR_ID).sum())
        print(
            f"[InVAR-clpretrain S1] C3 SECTOR selector ON: universe="
            f"{sector_universe_id} N={len(tickers)} coverage="
            f"{cov*100:.2f}% n_unknown={n_unknown} fold={cfg.fold}"
        )

    if pretrain_positive_method == "comovement" or is_joint:
        # A2 (2026-05-27): per-fold co-movement clusterer fit on the
        # TRAIN-segment daily-returns matrix only (no val/test leakage).
        # Cluster ids are cached at
        # cache/pretrain_improvements/comovement/<universe>/foldF/
        # cluster_ids.parquet; if cached we reuse, else fit and persist.
        from src.models.pretrain_improvements.comovement_clustering import (
            CoMovementClusterer,
            CoMovementConfig,
            cluster_ids_path,
            cluster_size_summary,
            load_cluster_ids,
            map_tickers_to_cluster_ids,
            save_cluster_ids,
        )
        comove_universe = str(
            getattr(cfg, "pretrain_comovement_universe_id", "sp500")
        )
        comove_K = int(
            getattr(cfg, "pretrain_comovement_n_clusters", 8)
        )
        comove_window = int(
            getattr(cfg, "pretrain_comovement_window", 252)
        )
        comove_cache = cluster_ids_path(
            universe=comove_universe, fold=int(cfg.fold),
        )
        if comove_cache.exists():
            cached = load_cluster_ids(
                universe=comove_universe, fold=int(cfg.fold),
            )
            cluster_id_lookup = dict(
                zip(
                    cached["ticker"].astype(str),
                    cached["cluster_id"].astype(int),
                )
            )
            print(
                f"[InVAR-clpretrain S1] A2 COMOVEMENT selector: loaded "
                f"cached clusters from {comove_cache}"
            )
        else:
            # Build per-fold train-segment daily-returns DataFrame from
            # the panel's feature 0 (log returns). x_raw is the
            # un-standardised panel so f0 is the raw log-return series.
            train_rets = x_raw[train_idx, :, 0].astype(np.float64)
            # Active mask per (t, n): a ticker may have zero-fill rows
            # outside its tradable range. Convert those to NaN so the
            # correlation builder ignores them per-pair.
            train_mask_pan = tradable[train_idx]
            train_rets = np.where(
                train_mask_pan, train_rets, np.nan,
            )
            train_dates_pd = pd.to_datetime(np.asarray(dates)[train_idx])
            train_df = pd.DataFrame(
                train_rets,
                index=train_dates_pd,
                columns=list(tickers),
            )
            comove_cfg = CoMovementConfig(
                universe=comove_universe,
                n_clusters=comove_K,
                window=comove_window,
                seed=int(cfg.seed),
            )
            clusterer = CoMovementClusterer(comove_cfg)
            cluster_id_lookup = clusterer.fit(
                train_df, n_clusters=comove_K,
            )
            save_cluster_ids(
                cluster_ids=cluster_id_lookup,
                universe=comove_universe,
                fold=int(cfg.fold),
                n_clusters=comove_K,
                n_train_days=int(train_df.shape[0]),
                n_windows=int(clusterer.n_windows_),
                seed=int(cfg.seed),
            )
            print(
                f"[InVAR-clpretrain S1] A2 COMOVEMENT selector: fit + "
                f"cached clusters at {comove_cache}"
            )
        ticker_cluster_ids = map_tickers_to_cluster_ids(
            list(tickers), cluster_id_lookup,
        )
        n_unknown_c = int((ticker_cluster_ids < 0).sum())
        cluster_sizes = cluster_size_summary(cluster_id_lookup)
        comove_ids_t = torch.from_numpy(
            ticker_cluster_ids.astype(np.int64)
        ).to(device)
        if is_joint:
            # A4 JOINT: keep the co-movement ids in a SEPARATE tensor so
            # the sector ids loaded above are not clobbered. The batch
            # loop uses both cohort tensors for the two per-stock terms.
            ticker_comove_ids_t = comove_ids_t
        else:
            # A2 pure-comovement path: reuse the sector-id tensor
            # pathway since the per-day per-stock InfoNCE loop downstream
            # consumes ticker_sector_ids_t as the single cohort id
            # (sector or co-movement is interchangeable in A2).
            ticker_sector_ids_t = comove_ids_t
        print(
            f"[InVAR-clpretrain S1] A2 COMOVEMENT selector ON: "
            f"universe={comove_universe} N={len(tickers)} "
            f"K={comove_K} n_unknown={n_unknown_c} fold={cfg.fold} "
            f"cluster_sizes={cluster_sizes} joint={is_joint}"
        )

    W = cfg.temporal_window
    aux_reg_on = bool(getattr(cfg, "pretrain_aux_regression_head", False))
    aux_reg_w = float(
        getattr(cfg, "pretrain_aux_regression_weight", 0.0)
    )

    # ---- C2 (2026-05-27): optional masked-feature-modeling pretext.
    # When pretrain_method == "masked_feature" the InfoNCE objective is
    # SKIPPED entirely and Stage 1 trains the encoder + a small linear
    # decoder via masked-feature MSE. When pretrain_method is any of
    # the InfoNCE family ("infonce_kmeans" default, "infonce_hmm",
    # "infonce_sector") the canonical / B1 / C3 path runs unchanged.
    pretrain_method = str(
        getattr(cfg, "pretrain_method", "infonce_kmeans")
    ).lower()
    if pretrain_method not in (
        "infonce_kmeans", "infonce_hmm", "infonce_sector", "masked_feature",
    ):
        raise ValueError(
            "cfg.pretrain_method must be one of 'infonce_kmeans', "
            "'infonce_hmm', 'infonce_sector', 'masked_feature'; got "
            f"{pretrain_method!r}"
        )
    # Mutual-exclusion guard (audit found WARN that there was no guard
    # across the B1 / B2 / C3 / C2 flags). Composing C2's masked-feature
    # pretext with any InfoNCE-only feature flag (B2 aux head, B1 HMM
    # selector, C3 sector selector) would yield an undefined hybrid;
    # raise instead.
    if pretrain_method == "masked_feature":
        if aux_reg_on:
            raise ValueError(
                "C2 masked_feature pretrain is mutually exclusive with "
                "B2 pretrain_aux_regression_head; set the aux head "
                "False when pretrain_method='masked_feature'."
            )
        if pretrain_regime_method == "hmm":
            raise ValueError(
                "C2 masked_feature pretrain is mutually exclusive with "
                "B1 pretrain_regime_method='hmm'; the InfoNCE objective "
                "is skipped, so the HMM selector has no role."
            )
        if pretrain_positive_method in ("sector", "comovement", "joint"):
            raise ValueError(
                "C2 masked_feature pretrain is mutually exclusive with "
                f"per-stock selector pretrain_positive_method="
                f"{pretrain_positive_method!r}; the InfoNCE objective "
                "is skipped, so the per-stock selector has no role."
            )

    mfm_on = pretrain_method == "masked_feature"
    mfm_mask_ratio = float(getattr(cfg, "pretrain_mask_ratio", 0.15))
    if mfm_on:
        from src.models.pretrain_improvements.masked_feature_modeling \
            import masked_feature_loss, random_feature_mask
        print(
            f"[InVAR-clpretrain S1] C2 MASKED FEATURE pretrain ON: "
            f"mask_ratio={mfm_mask_ratio:.3f} F={Fdim} "
            f"(InfoNCE objective SKIPPED)"
        )

    model = TemporalEncoderContrastivePretrainer(
        n_features=Fdim,
        temporal_window=W,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        e_layers=cfg.temporal_e_layers,
        dropout=cfg.dropout,
        activation=cfg.activation,
        proj_dim=CL_PROJ_DIM,
        aux_regression_head=aux_reg_on,
        masked_feature_head=mfm_on,
    ).to(device)
    # A1 (2026-05-27): optional sequential-pretrain continuation. When
    # init_from_ckpt is True we load the encoder weights already saved
    # at ckpt_path into model.encoder before training; the projection
    # head and (optional) aux heads are freshly initialised. Default
    # False preserves the canonical fresh-init path byte-identically.
    if init_from_ckpt:
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"init_from_ckpt=True but ckpt not found at {ckpt_path}; "
                "run the prior stage first."
            )
        prior_ckpt = torch.load(ckpt_path, map_location=device)
        prior_enc_state = prior_ckpt["encoder_state_dict"]
        target_keys = set(model.encoder.state_dict().keys())
        ckpt_keys = set(prior_enc_state.keys())
        if target_keys != ckpt_keys:
            raise RuntimeError(
                "A1 sequential: prior-stage encoder key mismatch. "
                f"missing={sorted(target_keys - ckpt_keys)} "
                f"unexpected={sorted(ckpt_keys - target_keys)}"
            )
        model.encoder.load_state_dict(prior_enc_state, strict=True)
        a_name, a_param = next(iter(model.encoder.named_parameters()))
        assert torch.allclose(
            a_param.detach().cpu(),
            prior_enc_state[a_name].detach().cpu().to(a_param.dtype),
        ), f"A1 init_from_ckpt no-op: {a_name} unchanged after load"
        print(
            f"[InVAR-clpretrain S1] A1 init_from_ckpt: loaded "
            f"{len(ckpt_keys)} encoder tensors from {ckpt_path}"
        )
    if aux_reg_on:
        # B2: y is the standard next-day forward-return panel built by
        # v2_runner.build_panel; pass through to the train loop so the
        # auxiliary cs_mse head can be supervised.
        y_t_pre = torch.from_numpy(y).to(device)
        print(
            f"[InVAR-clpretrain S1] B2 aux regression head ON "
            f"(weight={aux_reg_w:.4f})"
        )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    # Days usable as anchors: in the pretrain corpus, with a full
    # lookback window and >=3 active tickers.
    valid_days = [
        int(t) for t in pretrain_idx
        if int(t) >= W - 1 and tradable[int(t)].sum() >= 3
    ]
    batch_days = max(2, int(CL_BATCH_DAYS))
    steps_per_epoch = max(1, len(valid_days) // batch_days)
    total_steps = pretrain_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda s: warmup_cosine_lr(
            s, cfg.warmup_steps, total_steps
        ),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )
    n_pos = max(1, int(np.ceil(CL_POS_FRAC * batch_days)))
    print(f"[InVAR-clpretrain S1] {len(valid_days)} usable train days, "
          f"batch={batch_days}, pos/anchor={n_pos}, tau={CL_TEMPERATURE}")

    # B1: precompute a (day_idx -> posterior row) lookup so the batch
    # loop can build the positive mask in posterior space without
    # re-running predict_proba per step. Only populated when
    # pretrain_regime_method == "hmm"; otherwise stays None and the
    # canonical L2-nearest-neighbour selector runs below.
    posteriors_lookup_t: torch.Tensor | None = None
    if pretrain_regime_method == "hmm":
        assert day_posteriors_train_only is not None
        # train_idx -> posterior row mapping. We need a tensor over
        # the SAME global day-index space the batch loop uses (the
        # batch entries are global day indices).
        n_states_eff = int(day_posteriors_train_only.shape[1])
        full = np.zeros((T, n_states_eff), dtype=np.float32)
        full[np.asarray(train_idx, dtype=np.int64)] = (
            day_posteriors_train_only.astype(np.float32)
        )
        posteriors_lookup_t = torch.from_numpy(full).to(device)
        print(
            f"[InVAR-clpretrain S1] B1 posterior lookup built: "
            f"shape={posteriors_lookup_t.shape}, dtype=float32"
        )

    model.train()
    for epoch in range(pretrain_epochs):
        t0 = time.time()
        rng = np.random.RandomState(cfg.seed + epoch)
        perm = rng.permutation(np.asarray(valid_days, dtype=np.int64))
        losses = []
        for b0 in range(0, len(perm) - batch_days + 1, batch_days):
            batch = perm[b0: b0 + batch_days]
            # ---- LEAKAGE ASSERT: every Stage-1 day index is in the
            # pretrain (train) corpus; never val/test. ----
            for _t in batch:
                if int(_t) not in pretrain_set:
                    raise RuntimeError(
                        f"LEAKAGE: Stage-1 used day {int(_t)} not in "
                        "pretrain_idx (train corpus)."
                    )

            # C2 PATH: masked-feature pretext; no regime fingerprint, no
            # cohort mask. Build per-day per-stock masks INSIDE the
            # per-day forward below; loss is MSE on masked positions.
            if mfm_on:
                with torch.amp.autocast(
                    "cuda", enabled=(device.type == "cuda")
                ):
                    mfm_loss_terms: list[Tensor] = []
                    for _t in batch:
                        t = int(_t)
                        m_np = tradable[t]
                        active_idx = np.flatnonzero(m_np)
                        active_t = torch.from_numpy(active_idx).to(
                            device
                        )
                        x_win = x_t[
                            t - W + 1: t + 1, active_t, :
                        ].transpose(0, 1)                       # (N, T, F)
                        # Take the LAST time step as the reconstruction
                        # target; mask only the last row so the encoder
                        # must use the lookback context to recover it.
                        last_row = x_win[:, -1, :]              # (N, F)
                        masked_last, mask_ind = random_feature_mask(
                            last_row, mask_ratio=mfm_mask_ratio,
                        )
                        x_win_masked = x_win.clone()
                        x_win_masked[:, -1, :] = masked_last
                        recon = model.reconstruct_masked_features(
                            x_win_masked
                        )                                       # (N, F)
                        # Skip days with zero masked positions (sampling
                        # corner case at small N * F * mask_ratio).
                        if mask_ind.sum() > 0:
                            mfm_loss_terms.append(
                                masked_feature_loss(
                                    recon, last_row, mask_ind,
                                )
                            )
                    if mfm_loss_terms:
                        mfm_loss = torch.stack(mfm_loss_terms).mean()
                    else:
                        mfm_loss = torch.zeros(
                            (), device=device, dtype=torch.float32,
                        )
                    total_loss = mfm_loss
                optim.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                losses.append(float(mfm_loss.item()))
                continue

            # Per-day standardised regime fingerprints for this batch.
            keys_b = torch.from_numpy(
                day_keys_z[batch]
            ).float().to(device)                               # (B, 14)
            with torch.no_grad():
                bb = keys_b.shape[0]
                eye = torch.eye(
                    bb, dtype=torch.bool, device=device
                )
                if pretrain_positive_method in ("sector", "comovement"):
                    # C3 / A2 PATH: per-day per-stock InfoNCE; the day-
                    # level mask is unused. We build per-day same-
                    # cohort (sector or co-movement cluster) masks
                    # inside the per-day forward below.
                    pos_mask = None
                elif posteriors_lookup_t is None:
                    # CANONICAL PATH (pretrain_regime_method=="kmeans"):
                    # positives = n_pos nearest in-batch days by L2 in
                    # standardised regime-fingerprint space (self
                    # excluded). BYTE-IDENTICAL to the canonical
                    # selector; do not edit this branch.
                    kd = torch.cdist(keys_b, keys_b)           # (B, B)
                    kd = kd.masked_fill(eye, float("inf"))
                    k = min(n_pos, bb - 1)
                    nn_idx = torch.topk(
                        kd, k=k, dim=1, largest=False
                    ).indices                                  # (B, k)
                    pos_mask = torch.zeros(
                        bb, bb, dtype=torch.bool, device=device
                    )
                    pos_mask.scatter_(1, nn_idx, True)
                    pos_mask = pos_mask & (~eye)
                else:
                    # B1 PATH (pretrain_regime_method=="hmm"):
                    # positives = in-batch pairs whose posterior cosine
                    # similarity >= hmm_positive_threshold. Self
                    # excluded. Anchors with no positive at the
                    # threshold are skipped inside _supcon_infonce_loss.
                    batch_idx_t = torch.as_tensor(
                        batch.astype(np.int64), device=device
                    )
                    post_b = posteriors_lookup_t[batch_idx_t]  # (B, K)
                    post_n = torch.nn.functional.normalize(
                        post_b, dim=1, eps=1e-12
                    )
                    sim = post_n @ post_n.t()                  # (B, B)
                    pos_mask = (sim >= float(
                        hmm_positive_threshold
                    )) & (~eye)

            with torch.amp.autocast(
                "cuda", enabled=(device.type == "cuda")
            ):
                z_list = []
                aux_terms = []
                sector_loss_terms: list[Tensor] = []
                comove_loss_terms: list[Tensor] = []
                for _t in batch:
                    t = int(_t)
                    m_np = tradable[t]
                    active_idx = np.flatnonzero(m_np)
                    active_t = torch.from_numpy(active_idx).to(device)
                    # (N_active, T, F) lookback window (SAME slicing as
                    # the finetune loop in run_split).
                    x_win = x_t[
                        t - W + 1: t + 1, active_t, :
                    ].transpose(0, 1)
                    if is_joint:
                        # A4 JOINT PATH: one encoder forward yields both
                        # the pooled day embedding (for the day-level
                        # regime InfoNCE) and the per-stock projections
                        # (for the two per-stock SupCon terms over the
                        # GICS sector cohort and the co-movement cluster
                        # cohort respectively).
                        assert ticker_sector_ids_t is not None
                        assert ticker_comove_ids_t is not None
                        z_day, z_stocks = model.day_and_stock_projections(
                            x_win
                        )
                        z_list.append(z_day)
                        n_act = active_t.shape[0]
                        eye_n = torch.eye(
                            n_act, dtype=torch.bool, device=device,
                        )
                        for cohort_ids_t, term_list in (
                            (ticker_sector_ids_t, sector_loss_terms),
                            (ticker_comove_ids_t, comove_loss_terms),
                        ):
                            ids_day = cohort_ids_t[active_t]
                            same = (ids_day[:, None] == ids_day[None, :])
                            known = (ids_day[:, None] >= 0) & (
                                ids_day[None, :] >= 0
                            )
                            pos = (same & known) & (~eye_n)
                            if bool(pos.any()):
                                term_list.append(
                                    _supcon_infonce_loss_per_day(
                                        z_stocks, pos, CL_TEMPERATURE,
                                    )
                                )
                    elif pretrain_positive_method in ("sector", "comovement"):
                        # C3 / A2 PATH: per-stock projections (no pool)
                        # and a per-day SupCon InfoNCE over same-cohort
                        # peers (cohort = GICS sector for C3, co-movement
                        # cluster for A2; same downstream code).
                        assert ticker_sector_ids_t is not None
                        z_stocks = model.per_ticker_projections(x_win)
                        sec_ids_day = ticker_sector_ids_t[active_t]
                        same = (sec_ids_day[:, None]
                                == sec_ids_day[None, :])
                        known = (sec_ids_day[:, None] >= 0) & (
                            sec_ids_day[None, :] >= 0
                        )
                        n_act = sec_ids_day.shape[0]
                        eye_n = torch.eye(
                            n_act, dtype=torch.bool, device=device,
                        )
                        sec_pos = (same & known) & (~eye_n)
                        # Skip days with no same-cohort pair at all.
                        if bool(sec_pos.any()):
                            sector_loss_terms.append(
                                _supcon_infonce_loss_per_day(
                                    z_stocks, sec_pos, CL_TEMPERATURE,
                                )
                            )
                        # Still produce a day-level z so the
                        # downstream optimiser graph is consistent with
                        # the rest of the loop; not used in the loss.
                        z_list.append(z_stocks.mean(dim=0))
                    elif aux_reg_on:
                        z_t_emb, scores = model.day_embedding_with_scores(
                            x_win
                        )
                        z_list.append(z_t_emb)
                        # B2 auxiliary supervised target: next-day return
                        # for the SAME active tickers. cs_mse_loss
                        # z-scores the target per day internally.
                        y_day = y_t_pre[t, active_t]
                        mask_ones = torch.ones_like(
                            y_day, dtype=torch.bool
                        )
                        aux_terms.append(
                            cs_mse_loss(scores, y_day, mask_ones)
                        )
                    else:
                        z_list.append(model.day_embedding(x_win))
                if is_joint:
                    # A4 JOINT: aggregate the day-level regime InfoNCE
                    # with both per-stock SupCon terms. Each missing term
                    # (no in-batch positive on a given epoch step)
                    # contributes zero rather than NaN.
                    z = torch.stack(z_list, dim=0)             # (B, proj)
                    l_regime = _supcon_infonce_loss(
                        z, pos_mask, CL_TEMPERATURE
                    )
                    if sector_loss_terms:
                        l_sector = torch.stack(sector_loss_terms).mean()
                    else:
                        l_sector = torch.zeros(
                            (), device=device, dtype=torch.float32,
                        )
                    if comove_loss_terms:
                        l_comove = torch.stack(comove_loss_terms).mean()
                    else:
                        l_comove = torch.zeros(
                            (), device=device, dtype=torch.float32,
                        )
                    w_r = float(
                        getattr(cfg, "pretrain_joint_weight_regime", 1.0)
                    )
                    w_s = float(
                        getattr(cfg, "pretrain_joint_weight_sector", 1.0)
                    )
                    w_c = float(
                        getattr(cfg, "pretrain_joint_weight_comove", 1.0)
                    )
                    cl_loss = (
                        w_r * l_regime + w_s * l_sector + w_c * l_comove
                    )
                    total_loss = cl_loss
                elif pretrain_positive_method in ("sector", "comovement"):
                    if sector_loss_terms:
                        cl_loss = torch.stack(sector_loss_terms).mean()
                    else:
                        cl_loss = torch.zeros(
                            (), device=device, dtype=torch.float32,
                        )
                    total_loss = cl_loss
                else:
                    z = torch.stack(z_list, dim=0)             # (B, proj)
                    cl_loss = _supcon_infonce_loss(
                        z, pos_mask, CL_TEMPERATURE
                    )
                    if aux_reg_on and aux_terms:
                        aux_loss = torch.stack(aux_terms).mean()
                        total_loss = cl_loss + aux_reg_w * aux_loss
                    else:
                        total_loss = cl_loss

            optim.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            )
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            losses.append(float(cl_loss.item()))
        dt = time.time() - t0
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        loss_tag = "mfm_mse" if mfm_on else "infonce"
        print(f"[InVAR-clpretrain S1] epoch {epoch}: "
              f"{loss_tag}={mean_loss:.5f} "
              f"({len(losses)} batches, {dt:.1f}s)")

    # ---- Save the temporal-encoder state_dict ONLY (the projection
    # head is discarded). SAME key/path convention as the masked-recon
    # trainer so Stage-2's strict load works unchanged. ----
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fold": cfg.fold,
            "seed": cfg.seed,
            "pretrain_epochs": pretrain_epochs,
            "panel_kind": cfg.panel_kind,
            "encoder_state_dict": model.encoder.state_dict(),
        },
        ckpt_path,
    )
    print(f"[InVAR-clpretrain S1] saved encoder ckpt -> {ckpt_path}")


def run_stage1_sequential_pretrain(
    cfg: InvarSTXV2Config,
    pretrain_epochs: int,
    device: torch.device,
    ckpt_path: Path,
) -> None:
    """A1 (2026-05-27): sequential multi-stage Stage-1 pretrain.

    Reads the stage list from ``cfg.pretrain_stages`` (default
    ``["regime"]`` = canonical single-stage path, byte-identical to a
    direct ``run_stage1_pretrain`` call). For each stage we set the
    selector flags on ``cfg`` and call ``run_stage1_pretrain``; the
    second and later stages pass ``init_from_ckpt=True`` so the
    backbone continues training from the previous stage's saved
    encoder. Each stage trains for ``pretrain_epochs`` epochs.

    Supported stages: ``"regime"`` (k-means-8 day-level InfoNCE) and
    ``"sector"`` (C3 per-stock same-sector InfoNCE).
    """
    stages = list(getattr(cfg, "pretrain_stages", ["regime"]))
    if not stages:
        raise ValueError("cfg.pretrain_stages must be a non-empty list.")
    for s in stages:
        if s not in ("regime", "sector", "comovement"):
            raise ValueError(
                "A1/A2 pretrain_stages entries must be 'regime', "
                f"'sector', or 'comovement'; got {s!r}"
            )
    # A3 (2026-05-27): per-stock SupCon stages (sector, comovement) may
    # now BOTH appear in one curriculum as long as each appears at most
    # once. Each stage swaps cfg.pretrain_positive_method before its
    # run_stage1_pretrain call, so the per-day per-stock loss path is
    # rebuilt against the correct cohort id source (sector ids for the
    # 'sector' stage, co-movement cluster ids for the 'comovement'
    # stage). Repeats are rejected as redundant.
    per_stock_stages = [s for s in stages if s in ("sector", "comovement")]
    if len(per_stock_stages) != len(set(per_stock_stages)):
        raise ValueError(
            "pretrain_stages per-stock stages ('sector', 'comovement') "
            f"must be distinct (no repeats); got {stages}"
        )
    # Single-stage path: byte-identical to a direct call so the
    # canonical pipeline is preserved when pretrain_stages == ["regime"].
    if len(stages) == 1:
        run_stage1_pretrain(cfg, pretrain_epochs, device, ckpt_path)
        return
    # Cache the original selector flags so we can restore them after
    # the multi-stage loop (the loop overwrites them per stage).
    orig_pos = str(getattr(cfg, "pretrain_positive_method", "regime"))
    orig_method = str(getattr(cfg, "pretrain_method", "infonce_kmeans"))
    comove_epochs = int(
        getattr(cfg, "pretrain_comovement_epochs", pretrain_epochs)
    )
    print(
        f"[InVAR-clpretrain S1] A1/A2 SEQUENTIAL pretrain: "
        f"stages={stages} epochs_per_stage={pretrain_epochs} "
        f"(comove_epochs={comove_epochs})"
    )
    try:
        for stage_i, stage in enumerate(stages):
            if stage == "regime":
                cfg.pretrain_positive_method = "regime"
                cfg.pretrain_method = "infonce_kmeans"
                stage_epochs = pretrain_epochs
            elif stage == "sector":
                cfg.pretrain_positive_method = "sector"
                cfg.pretrain_method = "infonce_sector"
                stage_epochs = pretrain_epochs
            elif stage == "comovement":
                # A2 (2026-05-27): the per-day per-stock SupCon path is
                # shared with C3; only the cohort id source differs
                # (sector vs co-movement cluster). The trainer's batch
                # loop dispatches off pretrain_positive_method, so we
                # set it here for Stage 1b and the cluster ids are
                # built inside run_stage1_pretrain.
                cfg.pretrain_positive_method = "comovement"
                cfg.pretrain_method = "infonce_sector"
                stage_epochs = comove_epochs
            init_from = stage_i > 0
            print(
                f"[InVAR-clpretrain S1] A1/A2 stage {stage_i + 1}/"
                f"{len(stages)}: '{stage}' "
                f"(init_from_ckpt={init_from} epochs={stage_epochs})"
            )
            run_stage1_pretrain(
                cfg, stage_epochs, device, ckpt_path,
                init_from_ckpt=init_from,
            )
    finally:
        cfg.pretrain_positive_method = orig_pos
        cfg.pretrain_method = orig_method


def _assert_pretrain_causal(
    pretrain_idx: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Hard leakage guard: pretrain corpus must be a subset of
    ``train_idx`` and disjoint from ``val_idx | test_idx``."""
    p = set(int(i) for i in np.asarray(pretrain_idx).tolist())
    tr = set(int(i) for i in np.asarray(train_idx).tolist())
    va = set(int(i) for i in np.asarray(val_idx).tolist())
    te = set(int(i) for i in np.asarray(test_idx).tolist())
    if not p.issubset(tr):
        raise RuntimeError(
            "LEAKAGE: pretrain corpus is NOT a subset of train_idx "
            f"({len(p - tr)} day(s) outside train_idx)."
        )
    if p & va:
        raise RuntimeError(
            f"LEAKAGE: pretrain corpus intersects val_idx "
            f"({len(p & va)} day(s))."
        )
    if p & te:
        raise RuntimeError(
            f"LEAKAGE: pretrain corpus intersects test_idx "
            f"({len(p & te)} day(s))."
        )
    print(f"[InVAR-clpretrain] LEAKAGE-CHECK OK: |pretrain|={len(p)} "
          f"subset of |train|={len(tr)}; "
          f"intersect(val)={len(p & va)} intersect(test)={len(p & te)}")


# ============================================================================
# STAGE 2: finetune the full BANKLESS InVAR (canonical harness path).
# BYTE-IDENTICAL to train_invar_pretrain_v2.run_stage2_finetune.
# ============================================================================


def run_stage2_finetune(
    cfg: InvarSTXV2Config,
    finetune_epochs: int,
    device: torch.device,
    ckpt_path: Path,
) -> None:
    """Finetune the canonical bankless InVAR with the pretrained
    temporal encoder loaded in and a layer-wise LR.

    The data / fold / eval body is byte-identical to
    ``train_invar_stx_v2.main`` (same v2_runner calls, same SWA EMA loop,
    same early-stop, same JSON schema). The ONLY differences are:
    (1) the pretrained encoder weights are loaded into
    ``model.temporal_encoder`` with a strict key match + assertion,
    (2) two AdamW param groups give the pretrained encoder 0.25x LR.
    """
    # Lazy-import the F2 listwise rank loss to break the circular import
    # with ``src.invar.canonical`` (see module-level NOTE).
    from src.invar.training.loss import listmle_loss  # noqa: F401
    # Option C (2026-05-26): differentiable Sharpe surrogate at L1
    # finetune. Lazy-import keeps the canonical (loss_config != "diff_sharpe")
    # path free of any soft-topk overhead.
    from src.invar.training.loss import soft_topk_relaxation
    # ResInVAR-RL Phase 2: optional CAE-Head target-residualization.
    # Lazy-import so the canonical (cae_head_enabled=False) path does
    # not pay an import cost when ResInVAR is not active.
    from src.models.resinvar.cae_head import CAEHead
    # Robust-InVAR-RL Phase 1: group-DRO + top-bottom loss. Lazy-import
    # so the canonical (use_group_dro=False, lambda_top_bottom=0.0) path
    # pays no import cost.
    from src.models.robust_invar_rl.group_dro_loss import (
        compute_top_bottom_loss,
        group_dro_step,
    )
    cfg.epochs = int(finetune_epochs)
    if cfg.swa_warmup_epochs >= cfg.epochs:
        cfg.swa_warmup_epochs = max(0, cfg.epochs - 1)

    set_seeds(cfg.seed)
    print(f"[InVAR-clpretrain S2] fold={cfg.fold} seed={cfg.seed} "
          f"device={device}")

    # ---- v2_runner data / fold / eval calls: BYTE-IDENTICAL to
    # train_invar_stx_v2.py (same args, same order). ----
    x_raw, y, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    print(f"[InVAR-clpretrain S2] panel: T={T} N={N} F={Fdim}")
    min_n = 25 if cfg.panel_kind == "djia30" else 50
    if N < min_n:
        raise RuntimeError(
            f"Panel too small (N={N}, expected >={min_n} for "
            f"panel_kind={cfg.panel_kind})"
        )

    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]
    hist60 = mm["history_valid_60d"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    print(f"[InVAR-clpretrain S2] fold {cfg.fold}: "
          f"train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)}")

    x = standardize_features(x_raw, tradable, train_idx)
    age_feat = build_age_features(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)

    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)

    W = cfg.temporal_window

    # ---- Day-level regime memory keys + values (verbatim from
    # train_invar_stx_v2.py). ----
    day_keys, _ = build_episode_keys(
        dates=dates, log_returns=x_raw[..., 0], mask=tradable,
        cfg=EpisodeKeyConfig(),
    )
    feature_idx = [0, 1, 5, 6]
    n_summary = 2 * len(feature_idx) + 1
    day_values = np.zeros(
        (len(dates), day_keys.shape[1] + n_summary), dtype=np.float32,
    )
    day_values[:, : day_keys.shape[1]] = day_keys
    for t in range(len(dates)):
        m = tradable[t]
        if m.sum() < 5:
            continue
        for j, fi in enumerate(feature_idx):
            v = x_raw[t, m, fi]
            day_values[t, day_keys.shape[1] + 2 * j] = float(np.mean(v))
            day_values[t, day_keys.shape[1] + 2 * j + 1] = float(np.std(v))
        day_values[t, -1] = float(m.sum()) / 250.0
    cfg.day_value_dim = day_values.shape[1]

    # ---- Macro input + macro-gate input (verbatim). ----
    if cfg.panel_kind == "lattice_native":
        macro_path = Path(cfg.universal_macro_duration_parquet)
    elif cfg.panel_kind == "nasdaq100":
        macro_path = Path(cfg.nasdaq100_macro_duration_parquet)
    elif cfg.panel_kind == "djia30":
        macro_path = Path(cfg.djia30_macro_duration_parquet)
    elif cfg.panel_kind == "biotech_nbi":
        macro_path = Path(cfg.biotech_nbi_macro_duration_parquet)
    elif cfg.panel_kind == "biotech_nbi_enriched":
        # Same NBI macro feed (date-keyed, not ticker-keyed).
        macro_path = Path(cfg.biotech_nbi_macro_duration_parquet)
    else:
        macro_path = Path(cfg.biotech_macro_duration_parquet)
    if not macro_path.exists():
        print("[InVAR-clpretrain S2] macro parquet missing; building...")
        build_macro_duration_features()
    macro = pd.read_parquet(macro_path)
    macro_arr, macro_cols, _ = standardize_macro_duration(
        macro, dates, train_idx,
    )
    print(f"[InVAR-clpretrain S2] macro features: {len(macro_cols)} dims")

    gate_indices = [macro_cols.index(c) for c in MACRO_GATE_COLS
                    if c in macro_cols]
    if len(gate_indices) != len(MACRO_GATE_COLS):
        missing = [c for c in MACRO_GATE_COLS if c not in macro_cols]
        print(f"[InVAR-clpretrain S2] WARN missing gate cols: {missing}")
    macro_gate_macro = macro_arr[:, gate_indices].astype(np.float32)
    avg_corr_idx = EPISODE_KEY_COLS.index("cs_avg_pairwise_corr_60d")
    cs_disp_idx = EPISODE_KEY_COLS.index("cs_dispersion")
    avg_corr = day_keys[:, avg_corr_idx].astype(np.float32)
    cs_disp = day_keys[:, cs_disp_idx].astype(np.float32)
    avg_corr_tr = avg_corr[train_idx]
    cs_disp_tr = cs_disp[train_idx]
    avg_corr_z = ((avg_corr - avg_corr_tr.mean())
                  / max(avg_corr_tr.std(), 1e-6)).astype(np.float32)
    cs_disp_z = ((cs_disp - cs_disp_tr.mean())
                 / max(cs_disp_tr.std(), 1e-6)).astype(np.float32)
    macro_gate_arr = np.concatenate(
        [macro_gate_macro, avg_corr_z[:, None], cs_disp_z[:, None]],
        axis=1,
    ).astype(np.float32)
    print(f"[InVAR-clpretrain S2] macro_gate input: "
          f"{macro_gate_arr.shape[1]} dims")

    # ---- Per-ticker duration input (verbatim). ----
    duration_indices = resolve_duration_indices(cfg.panel_kind)
    duration_panel_block = _gather_or_zero(x, duration_indices).astype(
        np.float32
    )
    if cfg.panel_kind == "lattice_native":
        betas_path = Path(cfg.universal_rolling_betas_parquet)
    elif cfg.panel_kind == "nasdaq100":
        betas_path = Path(cfg.nasdaq100_rolling_betas_parquet)
    elif cfg.panel_kind == "djia30":
        betas_path = Path(cfg.djia30_rolling_betas_parquet)
    elif cfg.panel_kind == "biotech_nbi":
        betas_path = Path(cfg.biotech_nbi_rolling_betas_parquet)
    elif cfg.panel_kind == "biotech_nbi_enriched":
        # Same NBI rolling betas (computed from NBI prices).
        betas_path = Path(cfg.biotech_nbi_rolling_betas_parquet)
    else:
        betas_path = Path(cfg.biotech_rolling_betas_parquet)
    if not betas_path.exists():
        print("[InVAR-clpretrain S2] rolling betas parquet missing; "
              "building...")
        build_rolling_betas()
        betas_path = Path(cfg.biotech_rolling_betas_parquet)
    betas_long = pd.read_parquet(betas_path)
    betas_tensor = betas_to_tensor(betas_long, dates, tickers)
    bt_train = betas_tensor[train_idx]
    train_mask = tradable[train_idx]
    betas_std = np.zeros_like(betas_tensor)
    for fi in range(betas_tensor.shape[-1]):
        vals = bt_train[..., fi][train_mask]
        if vals.size < 2:
            mu, sd = 0.0, 1.0
        else:
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            if sd < 1e-6:
                sd = 1.0
        betas_std[..., fi] = (betas_tensor[..., fi] - mu) / sd
    betas_std = (betas_std * tradable[..., None]).astype(np.float32)
    duration_input_full = np.concatenate(
        [duration_panel_block, age_feat, betas_std], axis=-1,
    ).astype(np.float32)
    duration_input_dim = duration_input_full.shape[-1]
    print(f"[InVAR-clpretrain S2] duration input dim: "
          f"{duration_input_dim}")

    # ---- Build the canonical BANKLESS InVAR EXACTLY as
    # train_invar_stx_v2.py (enable_retrieval_bank stays False). ----
    assert cfg.enable_retrieval_bank is False, (
        "Canonical InVAR is BANKLESS; enable_retrieval_bank must be "
        "False for the pretrain protocol."
    )
    model = InvarSTXModel(
        cfg,
        n_features=Fdim,
        day_key_dim=day_keys.shape[1],
        duration_input_dim=duration_input_dim,
        macro_input_dim=macro_arr.shape[1],
        macro_gate_in_dim=macro_gate_arr.shape[1],
    ).to(device)
    model.day_memory.populate(
        keys=day_keys, values=day_values,
        day_indices=np.arange(len(dates)),
        train_day_indices=train_idx,
    )
    model.day_memory.to(device)

    # ---- Load the fold's pretrained temporal-encoder weights with a
    # STRICT key match into model.temporal_encoder; assert it loaded. ----
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Pretrained encoder ckpt not found: {ckpt_path}. Run "
            "stage 1 (--pretrain_only) for this fold first."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    enc_state = ckpt["encoder_state_dict"]
    target_keys = set(model.temporal_encoder.state_dict().keys())
    ckpt_keys = set(enc_state.keys())
    if target_keys != ckpt_keys:
        raise RuntimeError(
            "Pretrained encoder key mismatch with model.temporal_encoder. "
            f"missing={sorted(target_keys - ckpt_keys)} "
            f"unexpected={sorted(ckpt_keys - target_keys)}"
        )
    incompat = model.temporal_encoder.load_state_dict(
        enc_state, strict=True
    )
    assert not incompat.missing_keys and not incompat.unexpected_keys, (
        f"strict load failed: {incompat}"
    )
    # Verify at least one parameter actually changed to the loaded value
    # (defends against a silent no-op load).
    a_name, a_param = next(iter(model.temporal_encoder.named_parameters()))
    assert torch.allclose(
        a_param.detach().cpu(),
        enc_state[a_name].detach().cpu().to(a_param.dtype),
    ), f"pretrained weights NOT loaded into temporal_encoder.{a_name}"
    print(f"[InVAR-clpretrain S2] loaded pretrained temporal encoder "
          f"({len(ckpt_keys)} tensors, fold={ckpt.get('fold')}, "
          f"strict key match OK) from {ckpt_path}")

    allowed_train = torch.from_numpy(train_idx).long().to(device)

    # ---- ResInVAR-RL Phase 2: optional CAE-Head for target
    # residualization. Off by default (cfg.cae_head_enabled=False), in
    # which case the next 80 lines reduce to a no-op and the canonical
    # path is byte-identical. When on, the supervised finetune target
    # becomes y_tilde = z(eps) instead of y = z(r). ----
    # ---- Robust-InVAR-RL Phase 1: load macro-regime labels for group-DRO.
    # The k-means-8 soft-prob cache lives at
    # <gdro_regime_cache_root>/<gdro_universe_id>/foldF/probs.parquet
    # (built by invar_rl.layer3_control.regime_probs.precompute_all). The
    # per-day argmax over the 8-dim probability vector is the group label.
    # Days not present in the cache fall back to group 0; that should be
    # rare since the cache covers the full panel by construction.
    gdro_active = bool(getattr(cfg, "use_group_dro", False))
    gdro_groups_per_day: np.ndarray | None = None
    gdro_n_groups = 8
    gdro_q_state: torch.Tensor | None = None
    if gdro_active:
        cache_root = Path(getattr(cfg, "gdro_regime_cache_root",
                                  "cache/dr_rl/regime_probs"))
        uid = str(getattr(cfg, "gdro_universe_id", "sp500"))
        probs_path = cache_root / uid / f"fold{cfg.fold}" / "probs.parquet"
        if not probs_path.exists():
            raise FileNotFoundError(
                f"[ERR] group-DRO requires {probs_path}; build via "
                f"invar_rl.layer3_control.regime_probs.precompute_all "
                f"(universe={uid}, fold={cfg.fold})."
            )
        probs_df = pd.read_parquet(probs_path)
        prob_cols = [c for c in probs_df.columns if c.startswith("prob_")]
        gdro_n_groups = len(prob_cols)
        argmax_labels = probs_df[prob_cols].to_numpy().argmax(axis=1).astype(np.int64)
        date_to_group = dict(zip(
            pd.to_datetime(probs_df["date"]).dt.normalize(),
            argmax_labels,
        ))
        gdro_groups_per_day = np.zeros(len(dates), dtype=np.int64)
        for di, d in enumerate(dates):
            key = pd.Timestamp(d).normalize()
            if key in date_to_group:
                gdro_groups_per_day[di] = int(date_to_group[key])
        gdro_q_state = (torch.ones(gdro_n_groups, device=device,
                                   dtype=torch.float32) / gdro_n_groups)
        print(
            f"[INFO] group-DRO ACTIVE: universe={uid} G={gdro_n_groups} "
            f"eta={cfg.eta_gdro} lambda_tb={cfg.lambda_top_bottom} "
            f"M={cfg.m_top_bottom} cache={probs_path}",
            flush=True,
        )
    # Per-epoch per-group loss EMA + q_state held in a mutable dict so
    # the run_split closure can update without needing nonlocal.
    GDRO_EMA_DECAY = 0.9
    gdro_state: dict = {
        "loss_ema": (
            torch.zeros(gdro_n_groups, device=device, dtype=torch.float32)
            if gdro_active else None
        ),
        "q": gdro_q_state,
        "epoch_group_counts": np.zeros(gdro_n_groups, dtype=np.int64),
        "epoch_tb_sum": 0.0,
        "epoch_tb_n": 0,
    }

    cae_active = bool(getattr(cfg, "cae_head_enabled", False))
    cae_head: nn.Module | None = None
    if cae_active:
        cae_head = CAEHead(
            feature_dim=Fdim,
            k_latent=int(cfg.cae_head_k_latent),
            hidden_width=int(cfg.cae_head_hidden_width),
            dropout=float(cfg.cae_head_dropout),
            ridge_lambda=float(cfg.cae_head_ridge_lambda),
            orthogonality_penalty_weight=float(cfg.cae_head_lambda_orth),
        ).to(device)
        print(
            f"[InVAR-clpretrain S2] CAEHead ACTIVE: k_latent="
            f"{cfg.cae_head_k_latent} lambda_rec={cfg.cae_head_lambda_rec} "
            f"lambda_orth={cfg.cae_head_lambda_orth} orth="
            f"{cfg.cae_head_orthogonality_penalty} lr={cfg.cae_head_lr}"
        )

    # ---- LAYER-WISE LR: pretrained temporal-encoder params at
    # 0.25 * base; everything else at base (two AdamW param groups).
    # Phase 2: append a third param group for CAE-Head at constant
    # LR 1e-3 (no schedule applied by the LambdaLR below thanks to
    # its multiplicative form on group-level base LRs). ----
    enc_param_ids = {
        id(p) for p in model.temporal_encoder.parameters()
    }
    enc_params = [p for p in model.parameters()
                  if id(p) in enc_param_ids]
    other_params = [p for p in model.parameters()
                    if id(p) not in enc_param_ids]
    base_lr = cfg.learning_rate
    param_groups = [
        {"params": other_params, "lr": base_lr},
        {"params": enc_params, "lr": 0.25 * base_lr},
    ]
    if cae_active and cae_head is not None:
        param_groups.append({
            "params": list(cae_head.parameters()),
            "lr": float(cfg.cae_head_lr),
        })
    optim = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=cfg.weight_decay,
    )
    print(f"[InVAR-clpretrain S2] layer-wise LR: encoder={0.25 * base_lr:.2e} "
          f"({len(enc_params)} tensors) other={base_lr:.2e} "
          f"({len(other_params)} tensors)")
    total_steps = cfg.epochs * max(1, len(train_idx))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda s: warmup_cosine_lr(
            s, cfg.warmup_steps, total_steps
        ),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    macro_arr_t = macro_arr.astype(np.float32)
    macro_gate_t = macro_gate_arr.astype(np.float32)

    # ResInVAR-RL Phase 2: per-epoch CAE-Head telemetry accumulators
    # mutated by run_split when cae_active=True. y/y_tilde pairs are
    # cached on-the-fly so the end-of-epoch Pearson correlation can be
    # computed across a held subset (the last 256 active-stock samples).
    cae_epoch_stats: dict = {}

    def _reset_cae_epoch_stats() -> None:
        cae_epoch_stats.clear()
        cae_epoch_stats["l_rec_sum"] = 0.0
        cae_epoch_stats["l_orth_sum"] = 0.0
        cae_epoch_stats["total_sum"] = 0.0
        cae_epoch_stats["n_batches"] = 0
        cae_epoch_stats["y_pairs"] = []
        cae_epoch_stats["ytilde_pairs"] = []

    # ---- Option C (2026-05-26): differentiable Sharpe surrogate state.
    # Active only when cfg.loss_config == "diff_sharpe". The surrogate
    # is a moving-Sharpe: at each training day t we compute the soft
    # top-K L/S portfolio return pr_t differentiably from the current
    # day's y_full + cs_target_full, then form the Sharpe of the most
    # recent diff_sharpe_batch_days portfolio returns (older days are
    # detached and act as variance anchors; only the current pr_t
    # carries gradients). This preserves the existing per-day optim.step
    # semantics and keeps memory bounded.
    diff_sharpe_active = (
        getattr(cfg, "loss_config", "cs_mse") == "diff_sharpe"
    )
    diff_sharpe_K = int(getattr(cfg, "diff_sharpe_K", 50))
    diff_sharpe_tau = float(getattr(cfg, "diff_sharpe_temperature", 0.1))
    diff_sharpe_w = float(getattr(cfg, "diff_sharpe_weight", 0.2))
    diff_sharpe_bdays = max(
        2, int(getattr(cfg, "diff_sharpe_batch_days", 16))
    )
    # Rolling buffer of recent detached pr_t scalars (length <= bdays-1).
    diff_sharpe_state: dict = {
        "buffer": [],  # list of detached scalar tensors
        "epoch_sharpe_sum": 0.0,
        "epoch_pr_sum": 0.0,
        "epoch_n": 0,
    }
    if diff_sharpe_active:
        print(
            f"[InVAR-clpretrain S2] diff_sharpe ACTIVE: K={diff_sharpe_K} "
            f"tau={diff_sharpe_tau} weight={diff_sharpe_w} "
            f"batch_days={diff_sharpe_bdays}",
            flush=True,
        )

    def _reset_diff_sharpe_epoch_stats() -> None:
        diff_sharpe_state["buffer"] = []
        diff_sharpe_state["epoch_sharpe_sum"] = 0.0
        diff_sharpe_state["epoch_pr_sum"] = 0.0
        diff_sharpe_state["epoch_n"] = 0

    def run_split(idx: np.ndarray, train_: bool):
        """Run one pass over ``idx`` days. Byte-identical body to
        train_invar_stx_v2.run_split."""
        model.train(train_)
        losses = []
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        emask = np.zeros((T, N), dtype=bool)
        for t in idx:
            t = int(t)
            if t < W - 1:
                continue
            m_np = tradable[t]
            if m_np.sum() < 3:
                continue
            active_idx = np.flatnonzero(m_np)
            active_t = torch.from_numpy(active_idx).to(device)
            x_win = x_t[t - W + 1: t + 1, active_t, :].transpose(0, 1)
            y_target_full = y_t[t]
            lmask_t = torch.from_numpy(loss_mask[t]).to(device)

            # ResInVAR-RL Phase 2: build residualized target y_tilde via
            # CAE-Head on (x_t, r_t, active_mask_t). The full y_target
            # then gets swapped to a residual vector with the same
            # tradable shape so the rest of the loss path is unchanged.
            # cs_mse_loss z-scores its target internally, so we feed the
            # raw eps and let the existing z-score apply, matching the
            # spec's "y_tilde = z(eps) over active stocks".
            l_rec_term = None
            l_orth_term = None
            y_residualized_full = None
            if cae_active and cae_head is not None and train_:
                # Last-timestep ticker features over active set.
                cae_x = x_t[t, active_t, :]
                cae_r = y_target_full[active_t]
                cae_mask = torch.ones(
                    active_t.shape[0], device=device, dtype=torch.bool,
                )
                cae_out = cae_head(cae_x, cae_r, cae_mask)
                eps_active = cae_out["eps"]
                y_residualized_full = torch.zeros_like(y_target_full)
                y_residualized_full[active_t] = eps_active
                l_rec_term = cae_head.reconstruction_loss(
                    eps_active, cae_mask,
                )
                if bool(getattr(cfg, "cae_head_orthogonality_penalty", True)):
                    l_orth_term = cae_head.orthogonality_penalty(
                        cae_out["B"], cae_mask,
                    )

            day_query_key = torch.from_numpy(
                day_keys[t]
            ).float().to(device)
            regime_scalars = model.day_memory.standardize_query(
                day_query_key
            )[[0, 9]].clone()
            if torch.isnan(regime_scalars).any():
                regime_scalars = torch.zeros(2, device=device)

            dur_in = torch.from_numpy(
                duration_input_full[t, active_idx]
            ).float().to(device)
            macro_in = torch.from_numpy(
                macro_arr_t[t]
            ).float().to(device)
            macro_gate_in = torch.from_numpy(
                macro_gate_t[t]
            ).float().to(device)

            with torch.amp.autocast(
                "cuda", enabled=(device.type == "cuda")
            ):
                y_hat_active = model(
                    x_win,
                    day_query_key=day_query_key,
                    query_day_idx=t,
                    allowed_day_indices=allowed_train,
                    regime_scalars=regime_scalars,
                    duration_input=dur_in,
                    macro_input=macro_in,
                    macro_gate_input=macro_gate_in,
                )
                y_full = torch.zeros(N, device=device,
                                     dtype=y_hat_active.dtype)
                y_full[active_t] = y_hat_active
                # ResInVAR-RL Phase 2: when CAE-Head is active during
                # training, the supervised target is the residual
                # y_tilde, NOT the raw y. Eval / val / test paths keep
                # the raw y so val/test IC remain comparable to canonical.
                if (cae_active and train_
                        and y_residualized_full is not None):
                    cs_target_full = y_residualized_full
                else:
                    cs_target_full = y_target_full
                cs_loss_val = cs_mse_loss(y_full, cs_target_full, lmask_t)
                # F2 listwise rank loss upgrade (2026-05-26): optional
                # Plackett-Luce ListMLE term on the same day's active
                # cross-section. Default loss_config="cs_mse" gives
                # byte-identical canonical behaviour.
                #
                # Option C compose (2026-05-26): loss_config="listmle_soft"
                # also fires the ListMLE branch but with default weights
                # cs_mse=0.7 and listmle=0.5 (softer than the F2 phase-2
                # weights cs_mse=0.3 / listmle=1.0). The sbatch can still
                # pass --listmle-weight / --cs-mse-weight to override.
                _loss_cfg = getattr(cfg, "loss_config", "cs_mse")
                if _loss_cfg in ("listmle", "listmle_soft"):
                    lmask_bool = lmask_t.bool()
                    lmle_term = listmle_loss(
                        y_full, cs_target_full, lmask_bool,
                    )
                    if _loss_cfg == "listmle_soft":
                        _cs_default = 0.7
                        _lmle_default = 0.5
                    else:
                        _cs_default = 0.3
                        _lmle_default = 1.0
                    cs_loss = (
                        float(getattr(cfg, "cs_mse_weight", _cs_default))
                        * cs_loss_val
                        + float(getattr(cfg, "listmle_weight", _lmle_default))
                        * lmle_term
                    )
                else:
                    cs_loss = cs_loss_val
                # ResInVAR-RL Phase 2: append CAE-Head auxiliary losses
                # (reconstruction + optional orthogonality). Both terms
                # carry gradients into the InVAR backbone via y_tilde
                # AND into the CAE-Head loadings; no detach barrier.
                if cae_active and train_ and l_rec_term is not None:
                    cs_loss = cs_loss + (
                        float(cfg.cae_head_lambda_rec) * l_rec_term
                    )
                    if l_orth_term is not None:
                        cs_loss = cs_loss + l_orth_term
                # Robust-InVAR-RL Phase 1: group-DRO reweighting +
                # top-bottom listwise margin. Active only during training
                # and only when use_group_dro is True; otherwise this
                # block is a no-op and the canonical path is unchanged.
                if (gdro_active and train_
                        and gdro_groups_per_day is not None
                        and gdro_state["loss_ema"] is not None
                        and gdro_state["q"] is not None):
                    g_t = int(gdro_groups_per_day[t])
                    loss_ema = gdro_state["loss_ema"]
                    cur = cs_loss.detach().to(loss_ema.dtype)
                    new_ema = loss_ema.clone()
                    new_ema[g_t] = (
                        GDRO_EMA_DECAY * loss_ema[g_t]
                        + (1.0 - GDRO_EMA_DECAY) * cur
                    )
                    gdro_state["loss_ema"] = new_ema
                    _, q_new = group_dro_step(
                        per_group_losses=new_ema,
                        q_state=gdro_state["q"],
                        eta=float(cfg.eta_gdro),
                    )
                    gdro_state["q"] = q_new
                    gdro_state["epoch_group_counts"][g_t] += 1
                    weight_g = float(
                        (q_new[g_t] * float(gdro_n_groups)).item()
                    )
                    cs_loss = weight_g * cs_loss
                    if float(cfg.lambda_top_bottom) > 0.0:
                        tb_term = compute_top_bottom_loss(
                            scores=y_full,
                            returns=cs_target_full,
                            mask=lmask_t,
                            M=int(cfg.m_top_bottom),
                        )
                        cs_loss = cs_loss + (
                            float(cfg.lambda_top_bottom) * tb_term
                        )
                        gdro_state["epoch_tb_sum"] += float(tb_term.detach().item())
                        gdro_state["epoch_tb_n"] += 1

                # Option C (2026-05-26): differentiable Sharpe surrogate.
                # Active only during training and only when loss_config
                # == "diff_sharpe". Compute pr_t = (long_w * r).sum()/K
                # - (short_w * r).sum()/K differentiably via soft top-K
                # on the day's active subset, then form a moving Sharpe
                # over the last diff_sharpe_batch_days pr values (older
                # entries detached, only current attached). Append the
                # -sharpe term to cs_loss at weight diff_sharpe_weight.
                if diff_sharpe_active and train_:
                    lmask_bool_ds = lmask_t.bool()
                    if lmask_bool_ds.any():
                        s_a = y_full[lmask_bool_ds]
                        # Use the raw target return (NOT z-scored)
                        # because Sharpe is on the return scale.
                        # y_target_full is already next-day return per
                        # construction in v2_runner.build_panel.
                        r_a = y_target_full[lmask_bool_ds]
                        n_a = int(s_a.numel())
                        if n_a >= 4:
                            K_eff = max(1, min(diff_sharpe_K, n_a))
                            long_w = soft_topk_relaxation(
                                s_a, K_eff, diff_sharpe_tau,
                            )
                            short_w = soft_topk_relaxation(
                                -s_a, K_eff, diff_sharpe_tau,
                            )
                            pr_t = (
                                (long_w * r_a).sum()
                                - (short_w * r_a).sum()
                            ) / float(K_eff)
                            buf = diff_sharpe_state["buffer"]
                            if len(buf) >= 1:
                                anchors = torch.stack(buf)
                                pr_vec = torch.cat(
                                    [anchors, pr_t.unsqueeze(0)],
                                    dim=0,
                                )
                                mean_pr = pr_vec.mean()
                                std_pr = pr_vec.std(unbiased=False)
                                sharpe = mean_pr / (std_pr + 1.0e-6)
                                ds_term = -sharpe
                                cs_loss = cs_loss + (
                                    diff_sharpe_w * ds_term
                                )
                                diff_sharpe_state["epoch_sharpe_sum"] += float(
                                    sharpe.detach().item()
                                )
                                diff_sharpe_state["epoch_n"] += 1
                            diff_sharpe_state["epoch_pr_sum"] += float(
                                pr_t.detach().item()
                            )
                            # Append current pr_t (detached) to buffer
                            # for future steps; cap at bdays - 1 anchors.
                            buf.append(pr_t.detach())
                            if len(buf) > diff_sharpe_bdays - 1:
                                buf.pop(0)

            if train_:
                optim.zero_grad()
                scaler.scale(cs_loss).backward()
                scaler.unscale_(optim)
                # Phase 2: clip CAE-Head params alongside the backbone
                # so the joint update is well-conditioned.
                _clip_params = list(model.parameters())
                if cae_active and cae_head is not None:
                    _clip_params = _clip_params + list(cae_head.parameters())
                torch.nn.utils.clip_grad_norm_(_clip_params, cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                _maybe_update_swa()
                # ResInVAR-RL Phase 2 telemetry: per-day CAE-Head losses
                # + raw vs residual target pairs for the end-of-epoch
                # y_tilde-vs-y Pearson correlation diagnostic.
                if (cae_active and l_rec_term is not None
                        and y_residualized_full is not None):
                    cae_epoch_stats["l_rec_sum"] += float(
                        l_rec_term.detach().item()
                    )
                    if l_orth_term is not None:
                        cae_epoch_stats["l_orth_sum"] += float(
                            l_orth_term.detach().item()
                        )
                    cae_epoch_stats["total_sum"] += float(
                        cs_loss.detach().item()
                    )
                    cae_epoch_stats["n_batches"] += 1
                    with torch.no_grad():
                        y_a = y_target_full[active_t].detach().float()
                        yt_a = y_residualized_full[active_t].detach().float()
                        cae_epoch_stats["y_pairs"].append(
                            y_a.cpu().numpy()
                        )
                        cae_epoch_stats["ytilde_pairs"].append(
                            yt_a.cpu().numpy()
                        )
            losses.append(float(cs_loss.item()))
            y_hat_all[t] = y_full.detach().float().cpu().numpy()
            emask[t] = loss_mask[t]
        return (float(np.mean(losses)) if losses else float("nan"),
                y_hat_all, emask)

    # ---- SWA EMA state (byte-identical, incl. is_floating_point guard). ----
    ema_state: dict[str, torch.Tensor] | None = None
    swa_epoch_ref = {"epoch": 0}

    def _maybe_update_swa() -> None:
        nonlocal ema_state
        if not cfg.use_swa or swa_epoch_ref["epoch"] < cfg.swa_warmup_epochs:
            return
        with torch.no_grad():
            sd = model.state_dict()
            if ema_state is None:
                ema_state = {k: v.detach().clone() for k, v in sd.items()}
            else:
                d = float(cfg.swa_decay)
                for k in ema_state:
                    cur = sd[k].detach()
                    if torch.is_floating_point(ema_state[k]):
                        ema_state[k].mul_(d).add_(cur, alpha=1.0 - d)
                    else:
                        ema_state[k].copy_(cur)

    def _eval_split(idx: np.ndarray):
        if cfg.use_swa and ema_state is not None:
            saved = {k: v.detach().clone()
                     for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state)
            res = run_split(idx, train_=False)
            model.load_state_dict(saved)
            return res
        return run_split(idx, train_=False)

    history: list = []
    best_val_ic = -1e9
    best_state = None
    patience = 0
    # ResInVAR-RL Phase 2: open metrics.jsonl when CAE-Head is active
    # so per-epoch telemetry (l_rec, l_orth, total, y_tilde-vs-y corr)
    # can be tailed downstream. Path: cae_head_metrics_dir if set, else
    # default to the per-cell layout outputs/resinvar_rl/<universe>/
    # resinvar_canonical/seed<seed>/fold<fold>/metrics.jsonl.
    cae_metrics_path: Path | None = None
    cae_metrics_fh = None
    if cae_active:
        if cfg.cae_head_metrics_dir:
            cae_metrics_dir = Path(cfg.cae_head_metrics_dir)
        else:
            cae_metrics_dir = (
                Path("outputs/resinvar_rl") / cfg.panel_kind
                / "resinvar_canonical" / f"seed{cfg.seed}"
                / f"fold{cfg.fold}"
            )
        cae_metrics_dir.mkdir(parents=True, exist_ok=True)
        cae_metrics_path = cae_metrics_dir / "metrics.jsonl"
        cae_metrics_fh = open(cae_metrics_path, "w")
        print(
            f"[InVAR-clpretrain S2] CAE metrics jsonl -> {cae_metrics_path}"
        )
    for epoch in range(cfg.epochs):
        t0 = time.time()
        swa_epoch_ref["epoch"] = epoch
        np.random.seed(cfg.seed + epoch)
        perm = np.random.permutation(train_idx)
        if cae_active:
            _reset_cae_epoch_stats()
        if diff_sharpe_active:
            _reset_diff_sharpe_epoch_stats()
        train_loss, _, _ = run_split(perm, train_=True)
        val_loss, val_yhat, val_mask = _eval_split(val_idx)
        val_metrics = evaluate_predictions(val_yhat, y, val_mask, age_days)
        dt = time.time() - t0
        improved = val_metrics["ic"] > best_val_ic + 1e-5
        # Phase 2: end-of-epoch CAE telemetry.
        cae_record: dict | None = None
        if cae_active and cae_metrics_fh is not None:
            n_b = max(1, int(cae_epoch_stats.get("n_batches", 1)))
            l_rec_mean = cae_epoch_stats.get("l_rec_sum", 0.0) / n_b
            l_orth_mean = cae_epoch_stats.get("l_orth_sum", 0.0) / n_b
            total_mean = cae_epoch_stats.get("total_sum", 0.0) / n_b
            y_arr = np.concatenate(
                cae_epoch_stats.get("y_pairs", [np.zeros(1, np.float32)])
            )
            yt_arr = np.concatenate(
                cae_epoch_stats.get("ytilde_pairs", [np.zeros(1, np.float32)])
            )
            if y_arr.size >= 2 and yt_arr.size >= 2:
                if y_arr.std() < 1e-12 or yt_arr.std() < 1e-12:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(y_arr, yt_arr)[0, 1])
            else:
                corr = float("nan")
            cae_record = {
                "phase": "train",
                "epoch": int(epoch),
                "l_rec": float(l_rec_mean),
                "l_orth": float(l_orth_mean),
                "total_loss": float(total_mean),
                "y_tilde_corr_y": float(corr),
                "val_ic": float(val_metrics["ic"]),
                "val_rank_ic": float(val_metrics["rank_ic"]),
                "n_batches": int(n_b),
                "time_sec": round(dt, 2),
            }
            cae_metrics_fh.write(json.dumps(cae_record) + "\n")
            cae_metrics_fh.flush()
            print(
                f"[InVAR-clpretrain S2] CAE epoch {epoch}: "
                f"l_rec={l_rec_mean:.5f} l_orth={l_orth_mean:.5f} "
                f"total={total_mean:.5f} corr(y_tilde,y)={corr:+.4f}"
            )
        print(f"[InVAR-clpretrain S2] epoch {epoch}: "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_ic={val_metrics['ic']:+.4f} ({dt:.1f}s)"
              + ("  *best*" if improved else ""))
        if diff_sharpe_active and diff_sharpe_state["epoch_n"] > 0:
            n_ds = max(1, int(diff_sharpe_state["epoch_n"]))
            sharpe_mean = (
                diff_sharpe_state["epoch_sharpe_sum"] / n_ds
            )
            pr_mean = (
                diff_sharpe_state["epoch_pr_sum"]
                / max(1, int(diff_sharpe_state["epoch_n"]))
            )
            print(
                f"[INFO] diff_sharpe epoch {epoch}: "
                f"sharpe_mean={sharpe_mean:+.4f} "
                f"pr_mean={pr_mean:+.6f} n={n_ds}",
                flush=True,
            )
        if gdro_active and gdro_state["q"] is not None:
            q_vec = gdro_state["q"].detach().cpu().numpy()
            ema_vec = (gdro_state["loss_ema"].detach().cpu().numpy()
                       if gdro_state["loss_ema"] is not None
                       else np.zeros(gdro_n_groups))
            cnt = gdro_state["epoch_group_counts"]
            tb_avg = (gdro_state["epoch_tb_sum"] / max(1, gdro_state["epoch_tb_n"]))
            print(
                f"[INFO] gdro epoch {epoch}: q={np.round(q_vec, 3).tolist()} "
                f"loss_ema={np.round(ema_vec, 4).tolist()} "
                f"counts={cnt.tolist()} tb_mean={tb_avg:.5f}",
                flush=True,
            )
            # Reset per-epoch counters (q and loss_ema persist).
            gdro_state["epoch_group_counts"] = np.zeros(
                gdro_n_groups, dtype=np.int64
            )
            gdro_state["epoch_tb_sum"] = 0.0
            gdro_state["epoch_tb_n"] = 0
        hist_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ic": val_metrics["ic"],
            "val_rank_ic": val_metrics["rank_ic"],
            "time_sec": round(dt, 2),
        }
        if gdro_active and gdro_state["q"] is not None:
            hist_entry["gdro"] = {
                "q": gdro_state["q"].detach().cpu().numpy().tolist(),
                "loss_ema": (
                    gdro_state["loss_ema"].detach().cpu().numpy().tolist()
                    if gdro_state["loss_ema"] is not None else []
                ),
            }
        if cae_record is not None:
            hist_entry["cae"] = cae_record
        history.append(hist_entry)
        if improved:
            best_val_ic = val_metrics["ic"]
            src_state = (
                ema_state if (cfg.use_swa and ema_state is not None)
                else model.state_dict()
            )
            best_state = {k: v.detach().cpu().clone()
                          for k, v in src_state.items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"[InVAR-clpretrain S2] early stop epoch {epoch} "
                      f"best_val_ic={best_val_ic:+.4f}")
                break

    if cfg.use_swa and ema_state is not None:
        final_state = {k: v.detach().cpu().clone()
                       for k, v in ema_state.items()}
        print("[InVAR-clpretrain S2] SWA: using final EMA state for test")
    elif best_state is not None:
        final_state = best_state
    else:
        final_state = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}
    model.load_state_dict(final_state)

    # Optional: persist the full finetuned model state_dict alongside
    # the result JSON, gated by env var INVAR_SAVE_FULL_STATE=1 so the
    # canonical headline path (no env var) stays byte-identical. Used
    # by downstream consumers (e.g., InVAR-RL stages 2/3) that need to
    # load the trained layer-1 model and not just the encoder ckpt.
    if os.environ.get("INVAR_SAVE_FULL_STATE", "0") == "1":
        full_dir = Path(cfg.output_dir) / "_ckpt"
        full_dir.mkdir(parents=True, exist_ok=True)
        full_path = full_dir / f"fold{cfg.fold}_seed{cfg.seed}_full.pt"
        torch.save(
            {
                "state_dict": final_state,
                "fold": cfg.fold,
                "seed": cfg.seed,
                "panel_kind": cfg.panel_kind,
                "panel_end": cfg.panel_end,
                "two_regime_val": cfg.two_regime_val,
                "day_key_dim": int(day_keys.shape[1]),
                "duration_input_dim": int(duration_input_dim),
                "macro_input_dim": int(macro_arr.shape[1]),
                "macro_gate_in_dim": int(macro_gate_arr.shape[1]),
                "n_features": int(Fdim),
                # F3 (cross-stock attention) flags: persisted so the
                # downstream Layer-2 wrapper / SAC eval loaders can
                # reconstruct the right architecture. Reads default to
                # False / 4 on legacy ckpts via .get().
                "cross_stock_attn": bool(
                    getattr(cfg, "cross_stock_attn", False)
                ),
                "cross_stock_heads": int(
                    getattr(cfg, "cross_stock_heads", 4)
                ),
            },
            full_path,
        )
        print(
            f"[InVAR-clpretrain S2] saved full state_dict -> {full_path}"
        )

    _, test_yhat, test_mask = run_split(test_idx, train_=False)
    test_metrics = evaluate_predictions(test_yhat, y, test_mask, age_days)
    # Phase 2: emit a final test-segment record to metrics.jsonl and
    # close the file handle.
    if cae_active and cae_metrics_fh is not None:
        test_record = {
            "phase": "test",
            "epoch": int(cfg.epochs),
            "test_ic": float(test_metrics["ic"]),
            "test_rank_ic": float(test_metrics["rank_ic"]),
            "test_ndcg10": float(test_metrics["ndcg10"]),
            "test_ndcg50": float(test_metrics["ndcg50"]),
        }
        cae_metrics_fh.write(json.dumps(test_record) + "\n")
        cae_metrics_fh.flush()
        cae_metrics_fh.close()
    val_metrics_final = evaluate_predictions(
        val_yhat, y, val_mask, age_days
    )

    print(f"[InVAR-clpretrain S2] TEST ic={test_metrics['ic']:+.4f} "
          f"rank_ic={test_metrics['rank_ic']:+.4f} "
          f"ndcg10={test_metrics['ndcg10']:.4f} "
          f"ndcg50={test_metrics['ndcg50']:.4f}")

    # ---- Disk-safe write. NO predictions npz. Same JSON schema keys as
    # train_invar_stx_v2 (history entries contain "epoch" so the sbatch
    # skip-if-done test passes). ----
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fold{cfg.fold}_seed{cfg.seed}.json"
    payload = {
        "fold": cfg.fold,
        "seed": cfg.seed,
        "model": "InVAR-clpretrain (v2 protocol)",
        "panel_T": int(T),
        "panel_N": int(N),
        "panel_F": int(Fdim),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "ic": test_metrics["ic"],
        "rank_ic": test_metrics["rank_ic"],
        "ndcg10": test_metrics["ndcg10"],
        "ndcg50": test_metrics["ndcg50"],
        "test_cohort_ic": test_metrics["cohort_ic"],
        "val_ic": val_metrics_final["ic"],
        "val_rank_ic": val_metrics_final["rank_ic"],
        "val_cohort_ic": val_metrics_final["cohort_ic"],
        "history": history,
        "config": asdict(cfg),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[InVAR-clpretrain S2] wrote {out_path} (no npz; disk-safe)")


def main() -> None:
    """CLI entry point. Two-stage contrastive-pretrain -> finetune for
    canonical BANKLESS InVAR."""
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5],
                   required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--panel_kind", type=str, default="lattice_native",
                   choices=["biotech", "lattice_native", "nasdaq100",
                            "djia30", "biotech_nbi",
                            "biotech_nbi_enriched"])
    p.add_argument("--two_regime_val", action="store_true")
    p.add_argument("--output_dir", type=str,
                   default="results/invar_clpretrain")
    p.add_argument("--panel_end", type=str, default=None)
    p.add_argument("--pretrain_epochs", type=int, default=10)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--pretrain_only", action="store_true",
                   help="Run stage 1 (build+save ckpt) then exit.")
    p.add_argument("--skip_pretrain", action="store_true",
                   help="Load the existing ckpt; finetune only.")
    p.add_argument(
        "--pretrain-stages", "--pretrain_stages",
        dest="pretrain_stages",
        type=str, default="regime",
        help=(
            "A1/A2 (2026-05-27): comma-separated Stage-1 curriculum. "
            "'regime' (default) = canonical single-stage k-means-8 "
            "day-level InfoNCE (byte-identical). 'regime,sector' (A1) "
            "runs Stage 1a regime then continues training the SAME "
            "encoder under the C3 per-stock same-sector InfoNCE "
            "selector. 'regime,comovement' (A2) is the same shape with "
            "the cohort id sourced from a per-fold 252-day rolling "
            "correlation k-means cluster (data-driven, universe-"
            "agnostic). Only one of 'sector' / 'comovement' may appear "
            "per curriculum."
        ),
    )
    p.add_argument(
        "--pretrain-comovement-epochs",
        dest="pretrain_comovement_epochs",
        type=int, default=5,
        help=(
            "A2 (2026-05-27): epochs for the co-movement-clustered "
            "Stage 1b. Default 5 (spec range 5-10). Only consumed when "
            "pretrain_stages contains 'comovement'."
        ),
    )
    p.add_argument(
        "--pretrain-comovement-universe-id",
        dest="pretrain_comovement_universe_id",
        type=str, default="sp500",
        help=(
            "A2 cache key under cache/pretrain_improvements/"
            "comovement/<id>/foldF/cluster_ids.parquet. Default sp500."
        ),
    )
    p.add_argument(
        "--pretrain-comovement-n-clusters",
        dest="pretrain_comovement_n_clusters",
        type=int, default=8,
        help=(
            "A2 number of co-movement clusters (default 8 to match "
            "the k-means-8 regime cluster count)."
        ),
    )
    p.add_argument(
        "--pretrain-comovement-window",
        dest="pretrain_comovement_window",
        type=int, default=252,
        help=(
            "A2 rolling-correlation window length in trading days "
            "(default 252, one calendar year)."
        ),
    )
    args = p.parse_args()

    if args.pretrain_only and args.skip_pretrain:
        raise SystemExit(
            "--pretrain_only and --skip_pretrain are mutually exclusive."
        )

    # Default config = BANKLESS canonical (enable_retrieval_bank stays
    # at its False default; never enabled anywhere in this trainer).
    cfg = InvarSTXV2Config(fold=args.fold, seed=args.seed)
    cfg.panel_kind = args.panel_kind
    cfg.two_regime_val = args.two_regime_val
    cfg.output_dir = args.output_dir
    cfg.pretrain_stages = [
        s.strip() for s in str(args.pretrain_stages).split(",") if s.strip()
    ]
    cfg.pretrain_comovement_epochs = int(args.pretrain_comovement_epochs)
    cfg.pretrain_comovement_universe_id = str(
        args.pretrain_comovement_universe_id
    )
    cfg.pretrain_comovement_n_clusters = int(
        args.pretrain_comovement_n_clusters
    )
    cfg.pretrain_comovement_window = int(args.pretrain_comovement_window)
    if args.panel_end:
        cfg.panel_end = args.panel_end
    elif args.panel_kind == "lattice_native":
        cfg.panel_end = "2025-12-31"
    elif args.panel_kind == "nasdaq100":
        cfg.panel_end = "2025-12-31"
    elif args.panel_kind == "djia30":
        cfg.panel_end = "2025-12-31"
    elif args.panel_kind == "biotech_nbi":
        cfg.panel_end = "2025-12-31"
    elif args.panel_kind == "biotech_nbi_enriched":
        cfg.panel_end = "2025-12-31"
    assert cfg.enable_retrieval_bank is False, (
        "BANKLESS canonical invariant violated."
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[InVAR-clpretrain] fold={cfg.fold} seed={cfg.seed} "
          f"panel={cfg.panel_kind} two_regime_val={cfg.two_regime_val} "
          f"device={device}")

    # Fold-keyed encoder checkpoint path (one pretrain per fold; shared
    # across all finetune seeds for that fold).
    ckpt_dir = Path(cfg.output_dir) / "_ckpt"
    ckpt_path = ckpt_dir / f"fold{cfg.fold}_encoder.pt"

    if args.skip_pretrain:
        run_stage2_finetune(
            cfg, args.finetune_epochs, device, ckpt_path
        )
        return

    # Stage 1 contrastive pretrain (always seeded with cfg.seed; the
    # sbatch passes --seed 42 for the single per-fold pretrain). The
    # sequential wrapper preserves the canonical single-stage path
    # byte-identically when cfg.pretrain_stages == ["regime"].
    run_stage1_sequential_pretrain(
        cfg, args.pretrain_epochs, device, ckpt_path,
    )
    if args.pretrain_only:
        print("[InVAR-clpretrain] --pretrain_only: stage 1 done; "
              "exiting.")
        return

    # Single-process path: pretrain then finetune this (fold, seed).
    run_stage2_finetune(cfg, args.finetune_epochs, device, ckpt_path)


if __name__ == "__main__":
    main()
