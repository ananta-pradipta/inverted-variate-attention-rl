#!/bin/bash
#SBATCH --job-name=tli-stage2
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=stage2_decision_%j.out

# Phase 3 verification and run, all on Wulver (no local compute):
#   1. Full pytest, including the Stage 2 end-to-end differentiability test.
#   2. The Stage 2 sweep: pure-ranking control plus decision-focused
#      Variant A and Variant B, across the seed set and all folds,
#      warm-started from the Stage 1 checkpoints, producing the per-fold
#      comparison table.
#
# CPU only: the per-day QP solve (cvxpylayers + SCS) is the dominant cost
# and runs on CPU. Uses the dedicated project venv provisioned in Phase 2.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/three_layer_invar}"
VENV="${VENV:-$HOME/three_layer_invar-env}"
cd "$REPO_DIR"

module load python3 2>/dev/null || true
source "$VENV/bin/activate"
export PYTHONPATH="$REPO_DIR"

echo "Phase 3 pytest starting on $(hostname)"
python -m pytest -q

echo "Stage 2 sweep starting"
python -u -m src.training.stage2_decision \
  --base configs/base.yaml \
  --layer1 configs/layer1.yaml \
  --layer2 configs/layer2.yaml \
  --stage2 configs/stage2.yaml \
  --folds configs/folds.yaml

echo "Phase 3 complete"
