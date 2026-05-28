"""InVAR-RL Stage 1: train Layer 1 = canonical InVAR.

Stage 1 of the InVAR-RL three-layer build delegates to the canonical
InVAR pipeline at ``src.invar`` (bankless + regime-contrastive
pretrain, locked 2026-05-19; see ``docs/invar_headline_model.md`` for
the headline). This file is a thin wrapper: it iterates over (fold,
seed) cells, builds an :class:`src.invar.InVARConfig`, and calls
:func:`src.invar.train_invar` for each cell. The actual data, training
loop, evaluation, and checkpoint persistence are owned by the
canonical pipeline; this wrapper exists only to (1) integrate the
canonical pipeline into the InVAR-RL CLI surface and (2) write
per-(fold, seed) result JSONs alongside the existing InVAR-RL output
conventions.

The original Stage 1 script used the stripped InVAR skeleton in
``invar_rl.layer1_ranker.invar``; that file is kept for reference but
is no longer the training target.

Usage::

    python -m invar_rl.training.stage1_rank \
        --output-dir invar_rl/results/stage1 \
        --fold 1 --seed 42

Or via the Wulver sbatch ``invar_rl/scripts/wulver/invar_rl_stage1.sbatch``
which submits one job per fold and loops over seeds inside the job.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.invar import InVARConfig, train_invar


def _config_for(
    fold: int,
    seed: int,
    output_dir: str,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
    loss_config: str = "cs_mse",
    listmle_weight: float = 1.0,
    cs_mse_weight: float = 0.3,
    cross_stock_attn: bool = False,
    cross_stock_heads: int = 4,
    cae_head_enabled: bool = False,
    cae_head_k_latent: int = 3,
    cae_head_lambda_rec: float = 0.1,
    cae_head_lambda_orth: float = 0.01,
    cae_head_orthogonality_penalty: bool = True,
    cae_head_lr: float = 1.0e-3,
    cae_head_metrics_dir: str = "",
    use_group_dro: bool = False,
    eta_gdro: float = 0.05,
    lambda_top_bottom: float = 0.10,
    m_top_bottom: int = 50,
    gdro_universe_id: str = "sp500",
    diff_sharpe_weight: float = 0.2,
    diff_sharpe_K: int = 50,
    diff_sharpe_temperature: float = 0.1,
    diff_sharpe_batch_days: int = 16,
    pretrain_aux_regression_head: bool = False,
    pretrain_aux_regression_weight: float = 0.1,
    pretrain_regime_method: str = "kmeans",
    pretrain_hmm_n_states: int = 4,
    pretrain_hmm_positive_threshold: float = 0.7,
    pretrain_hmm_universe_id: str = "sp500",
    pretrain_positive_method: str = "regime",
    pretrain_sector_universe_id: str = "sp500",
    pretrain_method: str = "infonce_kmeans",
    pretrain_mask_ratio: float = 0.15,
    pretrain_stages: list[str] | None = None,
    pretrain_comovement_epochs: int = 5,
    pretrain_comovement_universe_id: str = "sp500",
    pretrain_comovement_n_clusters: int = 8,
    pretrain_comovement_window: int = 252,
    pretrain_joint_weight_regime: float = 1.0,
    pretrain_joint_weight_sector: float = 1.0,
    pretrain_joint_weight_comove: float = 1.0,
) -> InVARConfig:
    """Build the canonical InVAR config for one (fold, seed) cell.

    Matches the production sbatch in
    ``scripts/wulver/invar_clpretrain.sbatch``: bankless backbone,
    lattice_native panel, two-regime val protocol, panel_end pinned
    to 2025-12-31 for the universal S&P 500 panel.

    ``loss_config`` defaults to ``"cs_mse"`` for byte-identical
    canonical behaviour; pass ``"listmle"`` to add the F2 Plackett-Luce
    listwise rank-likelihood term to Stage-2 finetune.

    ``cross_stock_attn`` defaults to False for byte-identical canonical
    behaviour; pass True to enable the F3 explicit cross-stock
    self-attention block at the head boundary (2026-05-26 experiment).
    """
    cfg = InVARConfig(fold=fold, seed=seed)
    cfg.panel_kind = panel_kind
    cfg.two_regime_val = two_regime_val
    cfg.panel_end = panel_end
    cfg.output_dir = output_dir
    cfg.loss_config = loss_config
    # Option C: when 'listmle_soft' is selected and the per-flag weights
    # are still at the canonical 'listmle' defaults (cs_mse_weight=0.3,
    # listmle_weight=1.0), promote them to the Option C compose values
    # (cs_mse_weight=0.7, listmle_weight=0.5). Explicit user overrides
    # at the CLI flow through unchanged.
    if (loss_config == "listmle_soft"
            and abs(cs_mse_weight - 0.3) < 1e-9
            and abs(listmle_weight - 1.0) < 1e-9):
        cs_mse_weight = 0.7
        listmle_weight = 0.5
    cfg.listmle_weight = listmle_weight
    cfg.cs_mse_weight = cs_mse_weight
    cfg.cross_stock_attn = bool(cross_stock_attn)
    cfg.cross_stock_heads = int(cross_stock_heads)
    # ResInVAR-RL Phase 2: optional CAE-Head target-residualization.
    cfg.cae_head_enabled = bool(cae_head_enabled)
    cfg.cae_head_k_latent = int(cae_head_k_latent)
    cfg.cae_head_lambda_rec = float(cae_head_lambda_rec)
    cfg.cae_head_lambda_orth = float(cae_head_lambda_orth)
    cfg.cae_head_orthogonality_penalty = bool(cae_head_orthogonality_penalty)
    cfg.cae_head_lr = float(cae_head_lr)
    cfg.cae_head_metrics_dir = str(cae_head_metrics_dir)
    cfg.use_group_dro = bool(use_group_dro)
    cfg.eta_gdro = float(eta_gdro)
    cfg.lambda_top_bottom = float(lambda_top_bottom)
    cfg.m_top_bottom = int(m_top_bottom)
    cfg.gdro_universe_id = str(gdro_universe_id)
    # Option C (2026-05-26): differentiable Sharpe surrogate config.
    cfg.diff_sharpe_weight = float(diff_sharpe_weight)
    cfg.diff_sharpe_K = int(diff_sharpe_K)
    cfg.diff_sharpe_temperature = float(diff_sharpe_temperature)
    cfg.diff_sharpe_batch_days = int(diff_sharpe_batch_days)
    # B2 (2026-05-27): Stage-1 auxiliary next-day return regression head.
    cfg.pretrain_aux_regression_head = bool(pretrain_aux_regression_head)
    cfg.pretrain_aux_regression_weight = float(
        pretrain_aux_regression_weight
    )
    # B1 (2026-05-27): Stage-1 contrastive-positive selector mode.
    # "kmeans" preserves the canonical L2-nearest-neighbour selector
    # byte-identically; "hmm" enables the Gaussian HMM posterior
    # cosine-similarity selector in src/models/pretrain_improvements/
    # hmm_regime.py.
    method = str(pretrain_regime_method).lower()
    if method not in ("kmeans", "hmm"):
        raise ValueError(
            "--pretrain-regime-method must be 'kmeans' or 'hmm'; "
            f"got {pretrain_regime_method!r}"
        )
    cfg.pretrain_regime_method = method
    cfg.pretrain_hmm_n_states = int(pretrain_hmm_n_states)
    cfg.pretrain_hmm_positive_threshold = float(
        pretrain_hmm_positive_threshold
    )
    cfg.pretrain_hmm_universe_id = str(pretrain_hmm_universe_id)
    # C3 (2026-05-27): per-stock same-sector selector. "regime" is the
    # canonical / B1 day-level path (byte-identical when
    # pretrain_regime_method == "kmeans"); "sector" enables the per-day
    # per-stock SupCon InfoNCE over same-sector positives.
    # A4 (2026-05-27): "joint" aggregates the day-level regime InfoNCE
    # with both per-stock SupCon terms (sector + co-movement) in a single
    # Stage-1 objective. "comovement" is the A2 single-cohort per-stock
    # selector.
    pmethod = str(pretrain_positive_method).lower()
    if pmethod not in ("regime", "sector", "comovement", "joint"):
        raise ValueError(
            "--pretrain-positive-method must be 'regime', 'sector', "
            f"'comovement', or 'joint'; got {pretrain_positive_method!r}"
        )
    cfg.pretrain_positive_method = pmethod
    cfg.pretrain_sector_universe_id = str(pretrain_sector_universe_id)
    cfg.pretrain_joint_weight_regime = float(pretrain_joint_weight_regime)
    cfg.pretrain_joint_weight_sector = float(pretrain_joint_weight_sector)
    cfg.pretrain_joint_weight_comove = float(pretrain_joint_weight_comove)
    # C2 (2026-05-27): Stage-1 pretrain method selector. Default
    # "infonce_kmeans" composes with the canonical k-means selector;
    # "infonce_hmm" composes with B1; "infonce_sector" composes with
    # C3; "masked_feature" skips InfoNCE and runs BERT/MAE-style
    # masked-feature reconstruction. The trainer's mutual-exclusion
    # guard rejects "masked_feature" + (B1 hmm | B2 aux | C3 sector).
    pm = str(pretrain_method).lower()
    if pm not in (
        "infonce_kmeans", "infonce_hmm", "infonce_sector", "masked_feature",
    ):
        raise ValueError(
            "--pretrain-method must be one of 'infonce_kmeans', "
            "'infonce_hmm', 'infonce_sector', 'masked_feature'; got "
            f"{pretrain_method!r}"
        )
    cfg.pretrain_method = pm
    cfg.pretrain_mask_ratio = float(pretrain_mask_ratio)
    # A1/A2 (2026-05-27): sequential Stage-1 curriculum. Default
    # ["regime"] preserves the canonical single-stage path byte-
    # identically. ["regime", "sector"] is A1; ["regime", "comovement"]
    # is A2 (universe-agnostic data-driven cohort ids). A3 (2026-05-27):
    # ["regime", "sector", "comovement"] = the 3-stage curriculum that
    # composes both per-stock cohorts in sequence.
    stages = list(pretrain_stages) if pretrain_stages else ["regime"]
    for s in stages:
        if s not in ("regime", "sector", "comovement"):
            raise ValueError(
                "--pretrain-stages entries must be 'regime', 'sector', "
                f"or 'comovement'; got {s!r}"
            )
    # A3 allows BOTH per-stock stages, but each at most once and in a
    # fixed order is not required; only require they are DISTINCT (no
    # repeated sector or comove stage, which would be redundant).
    per_stock = [s for s in stages if s in ("sector", "comovement")]
    if len(per_stock) != len(set(per_stock)):
        raise ValueError(
            "--pretrain-stages per-stock stages ('sector', 'comovement') "
            f"must be distinct (no repeats); got {stages}"
        )
    cfg.pretrain_stages = stages
    # A2 (2026-05-27): co-movement clustering hyperparameters; only
    # consumed when 'comovement' is in the stage list.
    cfg.pretrain_comovement_epochs = int(pretrain_comovement_epochs)
    cfg.pretrain_comovement_universe_id = str(
        pretrain_comovement_universe_id
    )
    cfg.pretrain_comovement_n_clusters = int(pretrain_comovement_n_clusters)
    cfg.pretrain_comovement_window = int(pretrain_comovement_window)
    if cfg.enable_retrieval_bank:
        raise RuntimeError(
            "Canonical InVAR is bankless; enable_retrieval_bank must "
            "be False. The InVARConfig default should already enforce "
            "this; check src.invar.InVARConfig for drift."
        )
    return cfg


def run_one_cell(
    fold: int,
    seed: int,
    output_dir: str,
    panel_kind: str,
    panel_end: str,
    two_regime_val: bool,
    pretrain_epochs: int,
    finetune_epochs: int,
    pretrain_only: bool,
    skip_pretrain: bool,
    device: torch.device,
    loss_config: str = "cs_mse",
    listmle_weight: float = 1.0,
    cs_mse_weight: float = 0.3,
    cross_stock_attn: bool = False,
    cross_stock_heads: int = 4,
    cae_head_enabled: bool = False,
    cae_head_k_latent: int = 3,
    cae_head_lambda_rec: float = 0.1,
    cae_head_lambda_orth: float = 0.01,
    cae_head_orthogonality_penalty: bool = True,
    cae_head_lr: float = 1.0e-3,
    cae_head_metrics_dir: str = "",
    use_group_dro: bool = False,
    eta_gdro: float = 0.05,
    lambda_top_bottom: float = 0.10,
    m_top_bottom: int = 50,
    gdro_universe_id: str = "sp500",
    diff_sharpe_weight: float = 0.2,
    diff_sharpe_K: int = 50,
    diff_sharpe_temperature: float = 0.1,
    diff_sharpe_batch_days: int = 16,
    pretrain_aux_regression_head: bool = False,
    pretrain_aux_regression_weight: float = 0.1,
    pretrain_regime_method: str = "kmeans",
    pretrain_hmm_n_states: int = 4,
    pretrain_hmm_positive_threshold: float = 0.7,
    pretrain_hmm_universe_id: str = "sp500",
    pretrain_positive_method: str = "regime",
    pretrain_sector_universe_id: str = "sp500",
    pretrain_method: str = "infonce_kmeans",
    pretrain_mask_ratio: float = 0.15,
    pretrain_stages: list[str] | None = None,
    pretrain_comovement_epochs: int = 5,
    pretrain_comovement_universe_id: str = "sp500",
    pretrain_comovement_n_clusters: int = 8,
    pretrain_comovement_window: int = 252,
    pretrain_joint_weight_regime: float = 1.0,
    pretrain_joint_weight_sector: float = 1.0,
    pretrain_joint_weight_comove: float = 1.0,
) -> None:
    """Train canonical InVAR for one (fold, seed) cell."""
    cfg = _config_for(
        fold=fold,
        seed=seed,
        output_dir=output_dir,
        panel_kind=panel_kind,
        panel_end=panel_end,
        two_regime_val=two_regime_val,
        loss_config=loss_config,
        listmle_weight=listmle_weight,
        cs_mse_weight=cs_mse_weight,
        cross_stock_attn=cross_stock_attn,
        cross_stock_heads=cross_stock_heads,
        cae_head_enabled=cae_head_enabled,
        cae_head_k_latent=cae_head_k_latent,
        cae_head_lambda_rec=cae_head_lambda_rec,
        cae_head_lambda_orth=cae_head_lambda_orth,
        cae_head_orthogonality_penalty=cae_head_orthogonality_penalty,
        cae_head_lr=cae_head_lr,
        cae_head_metrics_dir=cae_head_metrics_dir,
        use_group_dro=use_group_dro,
        eta_gdro=eta_gdro,
        lambda_top_bottom=lambda_top_bottom,
        m_top_bottom=m_top_bottom,
        gdro_universe_id=gdro_universe_id,
        diff_sharpe_weight=diff_sharpe_weight,
        diff_sharpe_K=diff_sharpe_K,
        diff_sharpe_temperature=diff_sharpe_temperature,
        diff_sharpe_batch_days=diff_sharpe_batch_days,
        pretrain_aux_regression_head=pretrain_aux_regression_head,
        pretrain_aux_regression_weight=pretrain_aux_regression_weight,
        pretrain_regime_method=pretrain_regime_method,
        pretrain_hmm_n_states=pretrain_hmm_n_states,
        pretrain_hmm_positive_threshold=pretrain_hmm_positive_threshold,
        pretrain_hmm_universe_id=pretrain_hmm_universe_id,
        pretrain_positive_method=pretrain_positive_method,
        pretrain_sector_universe_id=pretrain_sector_universe_id,
        pretrain_method=pretrain_method,
        pretrain_mask_ratio=pretrain_mask_ratio,
        pretrain_stages=pretrain_stages,
        pretrain_comovement_epochs=pretrain_comovement_epochs,
        pretrain_comovement_universe_id=pretrain_comovement_universe_id,
        pretrain_comovement_n_clusters=pretrain_comovement_n_clusters,
        pretrain_comovement_window=pretrain_comovement_window,
        pretrain_joint_weight_regime=pretrain_joint_weight_regime,
        pretrain_joint_weight_sector=pretrain_joint_weight_sector,
        pretrain_joint_weight_comove=pretrain_joint_weight_comove,
    )
    ckpt_dir = Path(output_dir) / "_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold{fold}_encoder.pt"
    train_invar(
        cfg=cfg,
        ckpt_path=ckpt_path,
        device=device,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        pretrain_only=pretrain_only,
        skip_pretrain=skip_pretrain,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="InVAR-RL Stage 1: train Layer 1 = canonical InVAR."
    )
    p.add_argument(
        "--fold", type=int, required=True, choices=[1, 2, 3, 4, 5]
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--panel_kind",
        type=str,
        default="lattice_native",
        choices=["biotech", "lattice_native"],
    )
    p.add_argument("--panel_end", type=str, default="2025-12-31")
    p.add_argument("--two_regime_val", action="store_true", default=True)
    p.add_argument(
        "--output_dir",
        type=str,
        default="invar_rl/results/stage1",
    )
    p.add_argument("--pretrain_epochs", type=int, default=10)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--pretrain_only", action="store_true")
    p.add_argument("--skip_pretrain", action="store_true")
    p.add_argument(
        "--loss-config",
        type=str,
        default="cs_mse",
        choices=["cs_mse", "listmle", "listmle_soft", "diff_sharpe"],
        help=(
            "Stage-2 finetune loss. 'cs_mse' (default) keeps the canonical "
            "bankless+clpretrain behaviour byte-identical. 'listmle' adds "
            "a Plackett-Luce ListMLE rank-likelihood term on the same "
            "day's active cross-section (F2 fundamental L1 listwise rank "
            "loss upgrade, 2026-05-26). 'listmle_soft' is the Option C "
            "compose preset (2026-05-26): same ListMLE term but at weight "
            "0.5 (vs 1.0) and cs_mse weight 0.7 (vs 0.3), intended to "
            "compose with --cross-stock-attn (F3) and soften the F2 vs F5 "
            "trade-off observed in Phase 2. 'diff_sharpe' is the Option C "
            "differentiable-Sharpe surrogate (2026-05-26): adds a moving "
            "Sharpe term on the soft top-K L/S portfolio return on top of "
            "the existing cs_mse anchor; default weight 0.2 so the cs_mse "
            "core stays load-bearing. K and temperature tunable via the "
            "--diff-sharpe-* flags below."
        ),
    )
    p.add_argument(
        "--listmle-weight",
        type=float,
        default=1.0,
        help="Weight on the ListMLE term when --loss-config listmle.",
    )
    p.add_argument(
        "--cs-mse-weight",
        type=float,
        default=0.3,
        help="Weight on the cs_mse term when --loss-config listmle.",
    )
    p.add_argument(
        "--cross-stock-attn",
        action="store_true",
        help=(
            "F3 fundamental L1 upgrade (2026-05-26): enable the explicit "
            "cross-stock self-attention block at the head boundary, "
            "after day-memory fusion and before the score head. Off by "
            "default = canonical InVAR; on adds the MASTER-style "
            "stock-level attention layer."
        ),
    )
    p.add_argument(
        "--cross-stock-heads",
        type=int,
        default=4,
        help="Number of heads for the F3 cross-stock attention block.",
    )
    # ResInVAR-RL Phase 2 (2026-05-26): CAE-Head target-residualization
    # flags. Off by default; canonical InVAR path is byte-identical.
    p.add_argument(
        "--cae-head-enabled",
        action="store_true",
        help=(
            "Enable the ResInVAR-RL CAE-Head in Stage 2 finetune so the "
            "supervised target becomes y_tilde = z(eps) instead of "
            "y = z(r). Off by default = canonical InVAR."
        ),
    )
    p.add_argument(
        "--cae-head-k-latent", type=int, default=3,
        help="Number of latent factors in the CAE-Head.",
    )
    p.add_argument(
        "--cae-head-lambda-rec", type=float, default=0.1,
        help="Weight on the CAE-Head reconstruction loss.",
    )
    p.add_argument(
        "--cae-head-lambda-orth", type=float, default=0.01,
        help="Weight on the CAE-Head orthogonality penalty.",
    )
    p.add_argument(
        "--cae-head-orthogonality-penalty",
        action="store_true", default=True,
        help="Enable the CAE-Head orthogonality penalty (default on).",
    )
    p.add_argument(
        "--cae-head-no-orthogonality-penalty",
        action="store_false",
        dest="cae_head_orthogonality_penalty",
        help="Disable the CAE-Head orthogonality penalty.",
    )
    p.add_argument(
        "--cae-head-lr", type=float, default=1.0e-3,
        help="LR for the CAE-Head parameter group (no schedule).",
    )
    p.add_argument(
        "--cae-head-metrics-dir", type=str, default="",
        help=(
            "Directory for the CAE-Head metrics.jsonl file. Empty "
            "string defaults to outputs/resinvar_rl/<panel_kind>/"
            "resinvar_canonical/seed<seed>/fold<fold>/."
        ),
    )
    # Robust-InVAR-RL Phase 1 (2026-05-26).
    p.add_argument(
        "--use-group-dro", action="store_true", default=False,
        help=(
            "Enable Sagawa-style group-DRO reweighting of the Stage-2 "
            "hybrid loss over macro-regime labels (argmax of the "
            "k-means-8 cache at cache/dr_rl/regime_probs/<universe>/"
            "foldF/probs.parquet). Off by default = canonical InVAR."
        ),
    )
    p.add_argument(
        "--eta-gdro", type=float, default=0.05,
        help="DRO step size eta for the exponentiated-gradient q update.",
    )
    p.add_argument(
        "--lambda-top-bottom", type=float, default=0.10,
        help=(
            "Weight on the top-M / bottom-M pairwise margin loss "
            "(applied only when --use-group-dro is set)."
        ),
    )
    p.add_argument(
        "--m-top-bottom", type=int, default=50,
        help=(
            "Per-side M for the top-bottom loss (default 50; matches "
            "SP500 K=50 wrapper). NDX K=20 should pass 20; NBI K=25 "
            "should pass 25."
        ),
    )
    p.add_argument(
        "--gdro-universe-id", type=str, default="sp500",
        help=(
            "Cache key under cache/dr_rl/regime_probs/<id>/ for the "
            "group-DRO regime labels (e.g., 'sp500', 'nasdaq100', "
            "'biotech_nbi_enriched')."
        ),
    )
    # Option C (2026-05-26): differentiable Sharpe surrogate flags.
    # Active only when --loss-config diff_sharpe is set.
    p.add_argument(
        "--diff-sharpe-weight", type=float, default=0.2,
        help=(
            "Weight on the differentiable Sharpe surrogate term added "
            "to cs_loss. Default 0.2 (small additive regulariser, not a "
            "replacement for cs_mse)."
        ),
    )
    p.add_argument(
        "--diff-sharpe-k", type=int, default=50,
        help=(
            "Per-side soft top-K wrapper size for the L/S portfolio "
            "return that the Sharpe surrogate scores. Default 50 (SP500 "
            "wrapper); set 20 for NDX-100 and 25 for NBI-enriched."
        ),
    )
    p.add_argument(
        "--diff-sharpe-temperature", type=float, default=0.1,
        help=(
            "Sigmoid temperature for the soft top-K relaxation in "
            "score units. Smaller = sharper. Default 0.1."
        ),
    )
    p.add_argument(
        "--diff-sharpe-batch-days", type=int, default=16,
        help=(
            "Rolling window length (in training days) over which the "
            "Sharpe std is computed. The most recent day carries "
            "gradient; earlier days are detached anchors. Default 16."
        ),
    )
    # B2 (2026-05-27): Stage-1 auxiliary next-day return regression head.
    p.add_argument(
        "--pretrain-aux-regression-head",
        action="store_true",
        default=False,
        help=(
            "B2 (2026-05-27): add a small nn.Linear(d_model, 1) "
            "auxiliary head on top of the Stage-1 pretrain backbone "
            "and supervise it with cs_mse(score, next_day_return). "
            "Off by default = canonical InVAR Stage-1 (byte-identical)."
        ),
    )
    p.add_argument(
        "--pretrain-aux-regression-weight",
        type=float,
        default=0.1,
        help=(
            "Weight on the B2 auxiliary cs_mse term; total Stage-1 "
            "loss = InfoNCE + weight * cs_mse. Default 0.1 keeps the "
            "InfoNCE primary load-bearing."
        ),
    )
    # B1 (2026-05-27): Stage-1 contrastive-positive selector mode.
    p.add_argument(
        "--pretrain-regime-method",
        type=str,
        default="kmeans",
        choices=["kmeans", "hmm"],
        help=(
            "B1 (2026-05-27): Stage-1 contrastive-positive selector. "
            "'kmeans' (default) preserves the canonical L2 nearest-"
            "neighbour selector over the 14-d standardised episode-key "
            "fingerprint (byte-identical to canonical clpretrain). "
            "'hmm' fits a Gaussian HMM (hmmlearn preferred, sklearn "
            "GMM fallback) on the SAME TRAIN-day fingerprint and "
            "selects positives by posterior cosine similarity above "
            "--pretrain-hmm-positive-threshold."
        ),
    )
    p.add_argument(
        "--pretrain-hmm-n-states",
        type=int,
        default=4,
        help=(
            "Number of latent regimes for the B1 HMM (default 4; "
            "spec range 3-5)."
        ),
    )
    p.add_argument(
        "--pretrain-hmm-positive-threshold",
        type=float,
        default=0.7,
        help=(
            "Cosine-similarity floor on HMM posteriors for B1 SupCon "
            "positives (default 0.7). Anchors with no in-batch pair "
            "crossing the floor are skipped (matches the canonical "
            "no-positive handling in _supcon_infonce_loss)."
        ),
    )
    p.add_argument(
        "--pretrain-hmm-universe-id",
        type=str,
        default="sp500",
        help=(
            "Universe id for the B1 posterior cache root at "
            "cache/pretrain_improvements/hmm_regime/<id>/foldF/. "
            "Match the gdro universe convention (sp500, nasdaq100, "
            "biotech_nbi_enriched)."
        ),
    )
    # C3 (2026-05-27): sector-aware per-stock InfoNCE selector.
    p.add_argument(
        "--pretrain-positive-method",
        type=str,
        default="regime",
        choices=["regime", "sector", "comovement", "joint"],
        help=(
            "C3 (2026-05-27): Stage-1 positive-selector granularity. "
            "'regime' (default) preserves the canonical / B1 day-level "
            "selector path (byte-identical when "
            "--pretrain-regime-method kmeans). 'sector' runs a per-day "
            "per-stock SupCon InfoNCE whose positives are same-day "
            "same-sector peers (negatives = same-day different-sector "
            "peers). 'comovement' (A2) is the same per-stock path but "
            "cohort ids come from the data-driven co-movement clusters. "
            "'joint' (A4) aggregates the day-level regime InfoNCE with "
            "BOTH per-stock SupCon terms (sector + co-movement) in a "
            "single Stage-1 objective; see --pretrain-joint-weight-*."
        ),
    )
    p.add_argument(
        "--pretrain-sector-universe-id",
        type=str,
        default="sp500",
        help=(
            "Universe id for the C3 sector cache "
            "cache/sector_labels/<id>.parquet (default 'sp500')."
        ),
    )
    # A4 (2026-05-27): joint multi-objective loss weights.
    p.add_argument(
        "--pretrain-joint-weight-regime",
        type=float, default=1.0,
        help=(
            "A4 (2026-05-27): weight on the day-level regime InfoNCE "
            "term in the joint objective. Default 1.0."
        ),
    )
    p.add_argument(
        "--pretrain-joint-weight-sector",
        type=float, default=1.0,
        help=(
            "A4 (2026-05-27): weight on the per-stock same-sector SupCon "
            "term in the joint objective. Default 1.0."
        ),
    )
    p.add_argument(
        "--pretrain-joint-weight-comove",
        type=float, default=1.0,
        help=(
            "A4 (2026-05-27): weight on the per-stock same-co-movement-"
            "cluster SupCon term in the joint objective. Default 1.0."
        ),
    )
    # C2 (2026-05-27): masked-feature-modeling pretrain method.
    p.add_argument(
        "--pretrain-method",
        type=str,
        default="infonce_kmeans",
        choices=[
            "infonce_kmeans", "infonce_hmm", "infonce_sector",
            "masked_feature",
        ],
        help=(
            "C2 (2026-05-27): Stage-1 pretrain method selector. "
            "'infonce_kmeans' (default) preserves the canonical InfoNCE "
            "+ k-means-8 day-level selector byte-identically. "
            "'infonce_hmm' composes with B1 (--pretrain-regime-method "
            "hmm). 'infonce_sector' composes with C3 (--pretrain-"
            "positive-method sector). 'masked_feature' SKIPS InfoNCE "
            "entirely and runs a BERT/MAE-style masked-feature "
            "reconstruction pretext (decoder discarded after Stage 1). "
            "Mutually exclusive with B1/B2/C3 when set to "
            "'masked_feature'; the trainer raises on conflict."
        ),
    )
    p.add_argument(
        "--pretrain-mask-ratio",
        type=float,
        default=0.15,
        help=(
            "C2 per-position Bernoulli mask probability on the last-"
            "step feature row of each stock's lookback window. Default "
            "0.15 (BERT default). Only used when --pretrain-method "
            "masked_feature."
        ),
    )
    # A1/A2 (2026-05-27): sequential Stage-1 pretrain curriculum.
    p.add_argument(
        "--pretrain-stages",
        type=str,
        default="regime",
        help=(
            "A1/A2 (2026-05-27): comma-separated Stage-1 curriculum. "
            "'regime' (default) preserves the canonical single-stage "
            "k-means-8 day-level InfoNCE pretrain byte-identically. "
            "'regime,sector' (A1) runs Stage 1a regime (10 epochs) then "
            "continues training the SAME encoder under the C3 per-stock "
            "same-sector InfoNCE selector (5-10 epochs). 'regime,"
            "comovement' (A2) is the same shape but the per-stock "
            "cohort id comes from a per-fold 252-day rolling correlation "
            "k-means cluster (universe-agnostic, no external sector "
            "lookup). A3 (2026-05-27): 'regime,sector,comovement' runs "
            "all three stages in sequence (each per-stock stage at most "
            "once)."
        ),
    )
    # A2 (2026-05-27): co-movement clustering hyperparameters; only
    # consumed when --pretrain-stages contains 'comovement'.
    p.add_argument(
        "--pretrain-comovement-epochs",
        type=int, default=5,
        help=(
            "A2 (2026-05-27): epochs for the co-movement Stage 1b. "
            "Default 5 (spec range 5-10)."
        ),
    )
    p.add_argument(
        "--pretrain-comovement-universe-id",
        type=str, default="sp500",
        help=(
            "A2 cache key under cache/pretrain_improvements/"
            "comovement/<id>/foldF/cluster_ids.parquet."
        ),
    )
    p.add_argument(
        "--pretrain-comovement-n-clusters",
        type=int, default=8,
        help=(
            "A2 number of co-movement clusters (default 8 to match "
            "the k-means-8 regime cluster count)."
        ),
    )
    p.add_argument(
        "--pretrain-comovement-window",
        type=int, default=252,
        help=(
            "A2 rolling-correlation window length in trading days "
            "(default 252)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.pretrain_only and args.skip_pretrain:
        raise SystemExit(
            "--pretrain_only and --skip_pretrain are mutually exclusive."
        )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(
        f"[InVAR-RL stage 1] fold={args.fold} seed={args.seed} "
        f"panel={args.panel_kind} two_regime_val={args.two_regime_val} "
        f"device={device}"
    )
    run_one_cell(
        fold=args.fold,
        seed=args.seed,
        output_dir=args.output_dir,
        panel_kind=args.panel_kind,
        panel_end=args.panel_end,
        two_regime_val=args.two_regime_val,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        pretrain_only=args.pretrain_only,
        skip_pretrain=args.skip_pretrain,
        device=device,
        loss_config=args.loss_config,
        listmle_weight=args.listmle_weight,
        cs_mse_weight=args.cs_mse_weight,
        cross_stock_attn=args.cross_stock_attn,
        cross_stock_heads=args.cross_stock_heads,
        cae_head_enabled=args.cae_head_enabled,
        cae_head_k_latent=args.cae_head_k_latent,
        cae_head_lambda_rec=args.cae_head_lambda_rec,
        cae_head_lambda_orth=args.cae_head_lambda_orth,
        cae_head_orthogonality_penalty=args.cae_head_orthogonality_penalty,
        cae_head_lr=args.cae_head_lr,
        cae_head_metrics_dir=args.cae_head_metrics_dir,
        use_group_dro=args.use_group_dro,
        eta_gdro=args.eta_gdro,
        lambda_top_bottom=args.lambda_top_bottom,
        m_top_bottom=args.m_top_bottom,
        gdro_universe_id=args.gdro_universe_id,
        diff_sharpe_weight=args.diff_sharpe_weight,
        diff_sharpe_K=args.diff_sharpe_k,
        diff_sharpe_temperature=args.diff_sharpe_temperature,
        diff_sharpe_batch_days=args.diff_sharpe_batch_days,
        pretrain_aux_regression_head=args.pretrain_aux_regression_head,
        pretrain_aux_regression_weight=args.pretrain_aux_regression_weight,
        pretrain_regime_method=args.pretrain_regime_method,
        pretrain_hmm_n_states=args.pretrain_hmm_n_states,
        pretrain_hmm_positive_threshold=args.pretrain_hmm_positive_threshold,
        pretrain_hmm_universe_id=args.pretrain_hmm_universe_id,
        pretrain_positive_method=args.pretrain_positive_method,
        pretrain_sector_universe_id=args.pretrain_sector_universe_id,
        pretrain_method=args.pretrain_method,
        pretrain_mask_ratio=args.pretrain_mask_ratio,
        pretrain_stages=[
            s.strip() for s in str(args.pretrain_stages).split(",")
            if s.strip()
        ],
        pretrain_comovement_epochs=args.pretrain_comovement_epochs,
        pretrain_comovement_universe_id=(
            args.pretrain_comovement_universe_id
        ),
        pretrain_comovement_n_clusters=(
            args.pretrain_comovement_n_clusters
        ),
        pretrain_comovement_window=args.pretrain_comovement_window,
        pretrain_joint_weight_regime=args.pretrain_joint_weight_regime,
        pretrain_joint_weight_sector=args.pretrain_joint_weight_sector,
        pretrain_joint_weight_comove=args.pretrain_joint_weight_comove,
    )


if __name__ == "__main__":
    main()
