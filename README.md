# Inverted-Variate Attention Ranker with an RL Exposure Controller (InVAR-RL)

**InVAR-RL** is a three-layer system for regime-robust cross-sectional equity
investing: an **In**verted-**Var**iate attention **ranker** (Layer 1) that
scores stocks, a sparse interpretable allocator (Layer 2) that turns scores
into a portfolio, and a reinforcement-learning exposure controller (Layer 3)
that modulates the portfolio's market exposure as a function of the macro
regime.

## Abstract

Most learned trading systems either rank stocks (and leave position sizing to
a fixed rule) or hand the whole problem to a reinforcement-learning allocator
that must simultaneously learn what to hold and how much. Both degrade under
regime stress. InVAR-RL separates the concerns. Layer 1 is an inverted-variate
attention ranker: features are treated as tokens (an iTransformer-style
inverted attention) and trained with a regime-contrastive objective so the
cross-sectional return ranking is stable across calm and stressed windows.
Layer 2 maps ranker scores to a sparse, interpretable long or long-short book.
Layer 3 is a soft-actor-critic controller that does **not** re-pick names;
it adjusts net exposure and leverage conditioned on a macro-regime state
(Option A: RL as a macro-aware exposure controller, not an allocator). On
three equity universes (a 600-ticker S&P 500 panel, NASDAQ-100, and the
NASDAQ Biotechnology index) under a leakage-audited five-fold macro-stratified
walk-forward protocol, InVAR-RL leads on risk-adjusted return (Sharpe) on every
universe, while remaining transparent about where higher-variance ranker books
book larger raw returns.

This repository is the public code mirror; paper drafts and internal design
documents are kept in a private repository.

## Architecture

InVAR-RL is a stack of three independently trained, composable layers.

1. **Layer 1 — Inverted-Variate Attention Ranker (InVAR).**
   Per-(day, ticker) features are projected into variate tokens and processed
   by an inverted attention backbone (attention across features, not across
   time steps), giving a cross-sectional return-rank score per stock per day.
   The ranker is **ticker-inductive** (no per-ticker embeddings) and is
   pre-trained with a regime-contrastive objective ("clpretrain") so the
   ranking generalises across regime shifts. It runs bankless (no retrieval
   memory bank) in the canonical configuration.

2. **Layer 2 — Sparse Interpretable Allocation (SIA).**
   Ranker scores are converted into portfolio weights. The default is a
   top-K long-only or dollar-neutral long-short book; the `layer2_*` packages
   also provide a quadratic-program allocator, an uncertainty-regularised
   variant, and the sparse interpretable allocator used in the headline runs.

3. **Layer 3 — RL Exposure Controller.**
   A soft-actor-critic (SAC) agent observes a compact macro-regime state and
   the current book and outputs an exposure or leverage scalar. It optimises a
   risk-adjusted (online Sharpe) reward and is constrained so that it controls
   *how much* the book is exposed to the market, not *which* names are held.

## Universes and protocol

- **Universes**: a 600-ticker point-in-time S&P 500 panel (2015-2025, ~500
  names active per day, 2,755 trading days, membership-change aware),
  NASDAQ-100, and the NASDAQ Biotechnology index (NBI).
- **Five-fold macro-stratified walk-forward**, each fold a distinct regime:
  - F1: COVID crash and recovery (test 2020)
  - F2: rate-stress rotation (test 2021-H2 to 2022-H1, the discriminative fold)
  - F3: post-shock and banking stress (test 2022-H2 to 2023-H1)
  - F4: AI mega-cap rally and Fed pause (test 2024)
  - F5: Fed-cut and post-election (test 2025-H2)
- 5-day embargo at every train/val and val/test boundary; a two-regime
  validation window (2017-H2 calm plus 2018-H2 vol-spike) for early stopping.
- 5 seeds (42-46) per cell under a leakage audit.

## Results

Risk-adjusted performance on the long-only protocol (5 folds x 5 seeds, mean):
Sharpe ratio (SR), portfolio return (PR = final equity minus one), and
annualised return (AR).

| Universe    | SR   | PR   | AR   |
|-------------|------|------|------|
| S&P 500     | 0.85 | 0.20 | 0.22 |
| NASDAQ-100  | 0.84 | 0.17 | 0.20 |
| NBI biotech | 0.62 | 0.08 | 0.23 |

InVAR-RL leads on Sharpe on all three universes. On the long-short pure-alpha
protocol, higher-variance ranker books (FactorVAE, MASTER) post larger raw
PR/AR on the two smaller universes; the exposure controller optimises
risk-adjusted return rather than raw return, which the layered design makes
explicit.

**Baselines** retuned on the same panels and protocol: ranker baselines
MASTER, FactorVAE, StockMixer, and DySTAGE (each evaluated under both a
long-only top-K and a dollar-neutral long-short book), and whole-stack
reinforcement-learning baselines FinRL (A2C) and StockFormer.

## Repository layout

```
invar_rl/            InVAR-RL system code
  layer1_ranker/     Layer 1: inverted-variate attention ranker
  layer2_sia/        Layer 2: sparse interpretable allocator (and q/ur/alloc variants)
  layer2_q/  layer2_ur/  layer2_alloc/
  layer3_control/    Layer 3: SAC exposure controller
  data/              panel builders, lattice bridge, fold definitions, dataset adapters
  baselines/         FinRL, StockFormer, DeepTrader, non-learning and exposure baselines
  training/          per-universe training and evaluation entry points
  evaluation/        IC / Rank-IC / NDCG / portfolio and Sharpe metrics
  common/  configs/  shared utilities and YAML experiment configs
  scripts/           rollups and Slurm (Wulver) batch scripts
src/                 shared research library
  invar/             canonical InVAR ranker and adapters
  baselines/         ranker baseline trainers (MASTER, FactorVAE, StockMixer, DySTAGE, ...)
  lattice/           panel and macro-feature builders
  models/            pretraining objectives and robust-RL components
configs/             top-level experiment configs
scripts/             top-level analysis and batch scripts
tests/               unit and integration tests
requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

# Train Layer 1 (inverted-variate ranker) on the S&P 500 panel, one cell
python -m src.baselines.train_invar_clpretrain_v2 \
    --fold 1 --seed 42 --panel_kind lattice_native --two_regime_val \
    --output_dir results/invar/sp500

# Evaluate a ranker baseline under the long-only top-K protocol
python -m invar_rl.training.baselines_long_only_eval \
    --baseline master --fold 1 --seed 42 --top-k 30
```

Experiments are configured by YAML under `invar_rl/configs/` and `configs/`;
long-running jobs are launched via the Slurm batch scripts under
`invar_rl/scripts/wulver/`.

## References

The ranker and whole-stack baselines reimplemented for comparison:

1. Gu et al. *DySTAGE: Dynamic Spatio-Temporal Attention Graph Embedding.* ICAIF 2024.
2. Fan and Shen. *StockMixer.* AAAI 2024.
3. Li et al. *MASTER: Market-Guided Stock Transformer.* AAAI 2024.
4. Duan et al. *FactorVAE.* AAAI 2022.
5. Liu et al. *FinRL.* (deep RL for trading) 2021.
6. Gao et al. *StockFormer.* IJCAI 2023.
