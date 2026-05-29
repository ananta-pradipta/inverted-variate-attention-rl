# InVAR-RL: Reinforcement Learning as a Macro-State-Conditional Information-Transfer Layer

**InVAR-RL** reframes the role of reinforcement learning (RL) in cross-sectional
equity portfolio management. The conventional approach lets a single RL policy
emit the full per-stock weight vector, so RL must simultaneously learn *what to
hold* and *how much*. Instead, InVAR-RL uses RL as a **macro-state-conditional
information-transfer layer**: a soft actor-critic controller constrained to a
single 1-D exposure scalar that decides only *how much* market exposure to take,
while alpha extraction (*what to hold*) is handed off to a separately pretrained
supervised ranker.

## Abstract

State-of-the-art deep-RL trading agents train a policy that emits the full
per-stock weight vector. InVAR-RL argues this misplaces where RL supplies
value, and instead uses RL as a macro-aware exposure controller. We introduce
a **two-layer architecture**: a supervised, regime-contrastively pretrained
ranker (Layer 1) extracts cross-sectional alpha, and a soft actor-critic
controller (Layer 2) emits a single scalar exposure conditioned on the macro
state, with a fixed parameter-free top-K rule linking the two layers. Across
three equity universes (S&P 500, NASDAQ-100, and a sector-concentrated biotech
panel) and five macro-stratified folds, InVAR-RL wins the long-only protocol on
every universe and stays robust through the 2021-22 rate-stress regime on which
whole-stack-RL portfolio-MDP agents collapse. A five-ablation study isolates
the load-bearing component and establishes the mechanism: the controller's
value flows through its **observation** rather than trajectory memory, and a
supervised exposure rule cannot replace RL. The regime-contrastive pretrain is
the source of the rate-stress robustness and transfers across universes.

## Architecture

InVAR-RL composes **two learned layers** joined by a fixed, parameter-free
wrapper. There is no gradient flow between the layers, and Layer 1 is frozen
while Layer 2 trains. At inference the cascade is:

```
ŝ_t = InVAR(x_t)            # Layer 1: per-stock scores
w_t^wrap = g(ŝ_t)           # fixed deterministic top-K equal-weight L/S wrapper
e_t = π(o_t)                # Layer 2: SAC scalar exposure in [0, 1.5]
w_t^final = e_t · w_t^wrap   # exposure-scaled book
```

1. **Layer 1 — Macro-State-Contrastive InVAR Ranker.**
   An **In**verted-**VAR**iate attention transformer (iTransformer family). It
   tokenises each per-stock T=60-day feature window into variate tokens (one
   token per feature), FiLM-conditions them on a daily macro-state vector,
   applies bidirectional self-attention over variate tokens followed by
   cross-stock attention, and reads three heads: a score head (per-stock
   rank score), an auxiliary regime classifier (8-cluster soft labels), and a
   macro encoder producing `m_t` for Layer 2. It is **ticker-inductive** (no
   per-ticker embeddings). The backbone is pretrained for 10 epochs with a
   **macro-state-contrastive InfoNCE objective** (positives are days in the
   same k-means macro-state cluster, fold-causal), then fine-tuned 10 epochs on
   cross-sectional MSE. This pretrain is the component that keeps the F2
   rate-stress fold non-negative.

2. **Fixed deterministic top-K wrapper.**
   A parameter-free top-K equal-weight long-short book (K=50 per side, gross
   exposure 1, net 0); for the long-only protocol it degenerates to a top-K
   equal-weight long-only book. No learnable parameters, so the wrapper cannot
   perform portfolio optimisation.

3. **Layer 2 — SAC Macro-Aware Exposure Controller.**
   A soft actor-critic agent whose observation `o_t` concatenates the Layer-1
   score dispersion, the wrapper's predicted volatility and effective position
   count, the Layer-1 macro encoding `m_t`, and the agent's own risk state
   (rolling realised vol, current drawdown, current exposure,
   days-since-macro-change). The action is a **single scalar exposure**
   `e_t ∈ [0, 1.5]` that scales the wrapper book. It optimises a path-dependent
   risk-sensitive reward `R_t = r_{t+1} − λ·max(0, DD_t − DD*)` (drawdown
   budget) with twin Q-critics and automatic entropy tuning, trained for 20,000
   steps. Because the action is 1-D, the controller provably cannot solve a
   portfolio-optimisation problem; its only architectural role is macro-aware
   exposure timing.

## Universes and protocol

All three universes are evaluated under one identical five-fold protocol.

| Universe        | Names | Sector breadth   | Span      | Days  |
|-----------------|-------|------------------|-----------|-------|
| S&P 500         | 600   | 11 GICS sectors  | 2015-2025 | 2,755 |
| NASDAQ-100      | 178   | tech-concentrated| 2014-2025 | 2,845 |
| Biotech (NBI)   | 351   | single sector    | 2014-2025 | 2,845 |

- **Five expanding-window macro-stratified folds**: F1 COVID 2020, F2
  rate-stress 2021-22 (the discriminative fold), F3 post-stress and banking
  stress 2022-23, F4 AI mega-cap rally 2024, F5 Fed-cut 2025-H2.
- Fixed two-segment validation window (2017-H2 calm + 2018-H2 vol-spike),
  shared across all folds; 5-day embargo at every train/val/test boundary.
- 5 seeds (42-46) per cell = 25 cells per universe; leakage audited at three
  points. Primary metric: **pooled daily annualised Sharpe**.

## Results

Long-only protocol (Table 6, pooled over 25 cells per universe). SR =
annualised Sharpe (primary), PR = mean per-fold cumulative return, AR =
annualised return.

| Method                         | S&P 500 SR | NASDAQ-100 SR | Biotech NBI SR |
|--------------------------------|-----------|---------------|----------------|
| FactorVAE (AAAI'22)            | 0.26      | 0.53          | 0.35           |
| MASTER (AAAI'24)              | 0.45      | 0.64          | 0.45           |
| StockMixer (AAAI'24)         | 0.28      | 0.52          | 0.39           |
| DySTAGE (ICAIF'24)           | 0.39      | 0.50          | 0.39           |
| FinRL (deep RL, ICAIF'20)    | 0.13      | 0.15          | 0.13           |
| StockFormer (IJCAI'23)       | 0.10      | 0.80          | 0.49           |
| **InVAR-RL (ours)**          | **0.85**  | **0.84**      | **0.62**       |

InVAR-RL wins the long-only Sharpe on all three universes. The edge is downside
management on the F2 rate-stress fold: on S&P 500 per-fold Sharpe (Figure 7b),
InVAR-RL draws down to only -0.31 on F2 while the whole-stack RL agents FinRL
(-1.15) and StockFormer (-1.34) collapse.

On the **long-short dollar-neutral** protocol (pure alpha, Table 7) InVAR-RL
leads S&P 500 (SR 1.03, driving F2 positive), but on the two smaller,
large-cap-concentrated universes the FactorVAE ranker books lead (SR 1.53
NASDAQ-100, 1.50 NBI). This is an acknowledged negative-transfer result: with
fewer names the cross-sectional ranking signal is sharper and the exposure
controller has less idiosyncratic dispersion to exploit.

**Layer-1 ranker alone** (Table 5) attains pooled rank IC 0.0278, comparable to
MASTER (0.0274) and FactorVAE (0.0259), and is the only model that keeps the F2
fold non-negative.

### Mechanism (five-ablation study, S&P 500, Table 8)

- Swapping only the Layer-2 controller (frozen ranker, fixed book): SAC **0.62**
  beats constant-full-exposure (0.52), a supervised exposure head (0.45), and a
  volatility-targeting heuristic (0.31) -> a supervised rule cannot replace RL.
- Full-stack component ablations: full InVAR-RL **0.94**; randomised Layer-1
  scores collapse it to 0.10 (the ranker is the load-bearing alpha source); a
  learned mean-variance QP wrapper *degrades* it to 0.85 (the fixed wrapper
  beats a learned optimiser); masking the Layer-2 observation collapses the lift
  to 0.58 (value flows through the observation channel, not trajectory memory);
  recurrent PPO (0.50) does not beat feedforward PPO/SAC.
- Removing the macro-state-contrastive pretrain drops Layer-1 portfolio Sharpe
  from 0.54 to 0.35 -> the pretrain is the source of F2 robustness.

## Repository layout

The paper describes two *learned* layers plus a fixed wrapper; the source tree
numbers the wrapper/allocation stage separately, so the mapping is:

```
invar_rl/
  layer1_ranker/     Paper Layer 1: inverted-variate attention ranker (InVAR)
  layer2_alloc/      Allocation layer: fixed top-K equal-weight wrapper
                     (topk_layer.py, the headline) and the learned mean-variance
                     QP ablation (qp_layer.py)
  layer2_q/          Quantile-CVaR SAC variant (InVAR-RL-Q ablation)
  layer2_sia/        Sparse-invariant-actor variant (InVAR-RL-SIA; an
                     SP500-conditional F2-robustness lever, NOT the headline)
  layer3_control/    Paper Layer 2: SAC macro-aware exposure controller
  data/              panel builders, lattice bridge, fold definitions, adapters
  baselines/         FinRL, StockFormer, DeepTrader, non-learning baselines
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

# Layer 1: train the macro-state-contrastive InVAR ranker (one S&P 500 cell)
python -m src.baselines.train_invar_clpretrain_v2 \
    --fold 1 --seed 42 --panel_kind lattice_native --two_regime_val \
    --output_dir results/invar/sp500

# Evaluate a ranker baseline under the long-only top-K wrapper
python -m invar_rl.training.baselines_long_only_eval \
    --baseline master --fold 1 --seed 42 --top-k 30
```

Experiments are configured by YAML under `invar_rl/configs/` and `configs/`;
long-running jobs are launched via the Slurm batch scripts under
`invar_rl/scripts/wulver/`.

## Baselines

Three families, all retuned on the identical panel and protocol:

- **Cross-sectional rankers** (each wrapped with the same fixed top-K rule):
  MASTER (AAAI'24), FactorVAE (AAAI'22), iTransformer (ICLR'24), StockMixer
  (AAAI'24), DySTAGE (ICAIF'24), MERA (WWW'25).
- **Whole-stack RL** (one policy emits the full weight vector): FinRL
  (PPO/A2C/DDPG), StockFormer, DeepTrader.
- **Non-learning**: buy-and-hold, equal-weight, momentum (12-2), reversal-1M,
  volatility-targeted market.

## References

Cross-sectional ranker baselines:

1. Li et al. (2024). "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting." AAAI.
2. Duan et al. (2022). "FactorVAE: A Probabilistic Dynamic Factor Model for Prediction and Risk Attribution." AAAI.
3. Liu et al. (2024). "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." ICLR.
4. Fan and Shen (2024). "StockMixer: A Simple yet Strong MLP-Based Architecture for Stock Price Forecasting." AAAI.
5. Gu et al. (2024). "DySTAGE: Dynamic Spatio-Temporal Attention Graph Embedding for Stock Ranking." ICAIF.
6. (MERA) (2025). "Mixture-of-Experts Retrieval-Augmented model for stock prediction." WWW.

Whole-stack reinforcement-learning portfolio systems:

7. Liu et al. (2021). "FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance." ACM ICAIF.
8. Gao et al. (2023). "StockFormer: Learning Hybrid Trading Machines with Predictive Coding." IJCAI.
9. Wang et al. (2021). "DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management with Market Conditions Embedding." AAAI.

Method and finance background:

10. Haarnoja et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." ICML.
11. DeMiguel, Garlappi, and Uppal (2009). "Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy?" Review of Financial Studies.
12. Hamilton (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." Econometrica.
