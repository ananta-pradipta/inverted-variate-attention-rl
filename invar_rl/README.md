# InVAR-RL: Three-Layer Macro-State-Aware Cross-Sectional Equity System

The InVAR-RL build (locked name 2026-05-19, formerly `three_layer_invar`)
is a three-layer architecture for cross-sectional equity investing. Each
layer uses a different learning paradigm and the separation between
paradigms is preserved exactly in code.

- Layer 1, Ranking. The canonical InVAR ranker (the bankless +
  macro-contrastive pretrain variant locked 2026-05-19) lives in
  ``src/invar/`` at the repo root and is imported into Layer 1 via the
  thin adapter ``invar_rl/layer1_ranker/canonical_invar.py``. The
  original InVAR skeleton (``invar_rl/layer1_ranker/invar.py``) is
  retained as the data-flow contract reference but is no longer the
  training target. Headline reproduction: 5-fold x 5-seed pooled
  rank IC +0.0284 on the universal S&P 500 ``lattice_native`` panel
  with two-macro-state val (see ``docs/invar_headline_model.md``).
- Layer 2, Allocation. A differentiable portfolio layer mapping the
  score vector to weights by a constrained optimisation, enabling
  decision-focused learning that backpropagates into Layer 1.
  Supervised, not reinforcement learning.
- Layer 3, Exposure control. A recurrent reinforcement learning
  agent that observes a compact, detached summary of Layers 1 and 2
  plus its own risk state and chooses exposure, trained with a
  path-dependent, risk-sensitive reward. This is the only
  reinforcement learning component.

Architectural invariant: there is no gradient flow from Layer 3 into
Layer 2 or Layer 1. Layer 1 and Layer 2 outputs enter Layer 3 as
detached observations, and during Layer 3 training the lower layers
are frozen with precomputed outputs.

## Branch state (2026-05-19)

This build originally lived on the orphan branch ``three-layer-invar``.
That branch is preserved as ``archive/three-layer-invar-2026-05-19`` and
no longer receives commits. Going forward, InVAR-RL development happens
on ``v3`` under this ``invar_rl/`` directory; the standalone canonical
InVAR is at ``src/invar/`` (re-exporting from
``src/baselines/train_invar_stx_v2.py`` and
``src/baselines/train_invar_clpretrain_v2.py``).

Migration state per stage:

- Standalone canonical InVAR: ``src/invar/`` -- DONE.
- InVAR-RL subtree (configs, layer 2, layer 3, evaluation, common
  utilities, panel loaders, sbatches): copied from
  ``archive/three-layer-invar-2026-05-19`` -- DONE.
- Layer 1 adapter ``CanonicalInVARLayer1``: scaffold only -- WIP.
- Data-pipeline bridge ``lattice_native`` -> 8 canonical InVAR inputs:
  PENDING (task #7).
- Stage 1/2/3 rewire to canonical InVAR adapter: PENDING (task #8).
- Wulver smoke run to verify +0.0284 reproduces under the new
  stage-1 wrapper: PENDING (task #9).

Until tasks #7-9 land, the in-tree InVAR skeleton
(``invar_rl/layer1_ranker/invar.py``) is what stages 2/3 currently
instantiate. The new ``CanonicalInVARLayer1`` is present but inert.

## Project status

Phase 0 (scaffolding and infrastructure) is implemented: configuration
system, data contract, synthetic data generator, walk-forward splitter, and
the per-day dataset. No model layers are implemented yet. Later-phase modules
are present as stubs.

## Environment setup

Python 3.11 or later is the target. Create a virtual environment and install
the pinned requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Phase 0 only requires numpy, pandas, scikit-learn, PyYAML, torch, and pytest.
The remaining packages (cvxpylayers, gymnasium, stable-baselines3,
sb3-contrib) are pinned for later phases.

## Running the Phase 0 tests

From the repository root:

```bash
cd three_layer_invar
PYTHONPATH=. pytest -q
```

## The data contract you must implement

The real data pipeline lives outside this repository. To plug in real data,
implement `src/data/contract.py:PanelDataContract`. The interface provides,
for any trading-day index t:

- The point-in-time set of active tickers on day t.
- For each active ticker, a lookback feature window of shape (L, F).
- The daily macro vector of fixed dimension F_macro.
- For each active ticker, the 5-day forward return label and its within-day
  cross-sectional z-scored version.
- A boolean tradable-and-labelled mask per ticker.
- The full ordered list of trading days and realised 1-day and 5-day forward
  returns per ticker (used by Layer 3, which replays realised returns).

All standardisation, covariance, and macro-state statistics must be computed from
training-fold data only with a real-time convention. No future, validation,
or test information may enter a training computation.

`src/data/synthetic.py:SyntheticPanel` is a fully seeded reference
implementation with injected latent macro-states. Every later phase depends only
on the contract, never on a raw file format, so development and testing
proceed end to end without the real dataset.

## Configuration

All hyperparameters, paths, fold definitions, and dimensions live in
`configs/` and are parsed into typed dataclasses by `src/common/config.py`.

- `base.yaml`: seeds, device, synthetic generator parameters, output paths.
- `folds.yaml`: walk-forward folds as inclusive day-index ranges, the global
  embargo, and the single out-of-distribution stress fold flag.
- `layer1.yaml`, `layer2.yaml`, `layer3.yaml`: defaults consumed by later
  phases; no Phase 0 code reads them.

## Conventions

- Reproducibility under the seed set {42, 43, 44, 45, 46}. Seed Python,
  NumPy, and PyTorch through `src/common/seeding.py`.
- No em-dashes, no en-dashes, no decorative icons anywhere.
- Implement only the current phase, then stop and summarise.
