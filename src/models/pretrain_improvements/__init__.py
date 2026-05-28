"""Pretrain-improvement experiments on top of canonical InVAR Stage 1.

Each module here is a candidate modification to the canonical bankless
regime-contrastive (clpretrain) Stage 1. Modules are written so the
canonical path is byte-identical when the corresponding feature flag
is off, and the experimental path is opt-in via an explicit CLI flag
on ``invar_rl.training.stage1_rank``.

Modules:
  - ``hmm_regime``: B1 (2026-05-27) Gaussian HMM regime labeler. Fits
    a temporal-Gaussian HMM (via ``hmmlearn`` when available, GMM
    fallback) on the same 14-d regime fingerprint that the canonical
    L2-nearest-neighbour positive selector consumes, and exposes a
    per-day posterior over ``n_states`` latent regimes for use as the
    InfoNCE positive selector via cosine similarity over posteriors.
  - ``sector_positives``: C3 (2026-05-27) sector-aware InfoNCE
    selector. Replaces the day-level regime/HMM selector with a per-
    stock InfoNCE whose positives are same-day same-sector peers and
    negatives are same-day different-sector peers. Sector ids come
    from ``cache/sector_labels/<universe>.parquet`` (SP500 = 11 GICS
    sectors; auto-built from ``data/processed/sp500_sector_map.csv``
    plus a small hard-coded supplement for recent index additions).
  - ``masked_feature_modeling``: C2 (2026-05-27) masked feature
    reconstruction pretext. Per stock per day a Bernoulli mask
    (default 15%) over the F feature channels is applied to the LAST-
    step row of the lookback; a small linear decoder
    ``d_model -> feature_dim`` reconstructs the masked feature values
    from the encoder output. Loss is MSE on masked positions only.
    Different inductive bias from InfoNCE: dense feature dependencies
    rather than cohort coherence. Head is discarded after Stage 1; only
    the encoder is carried into Stage 2.
  - ``comovement_clustering``: A2 (2026-05-27) co-movement-clustered
    per-stock InfoNCE positives for sequential Stage 1 (regime ->
    comovement). Per fold, the TRAIN-segment daily returns matrix is
    aggregated into a single 252-day rolling correlation matrix; the
    distance ``d = 1 - rho`` is k-means-clustered with K=8 so each
    ticker carries a data-driven co-movement cluster id. Sequential
    composition keeps the regime InfoNCE in Stage 1a; Stage 1b then
    runs a per-day per-stock SupCon InfoNCE whose positives are same-
    day same-co-movement-cluster peers (universe-agnostic, complements
    C3 sector positives which fail on sector-degenerate universes).
"""
