#!/bin/bash
#SBATCH --job-name=tli-phase6
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=phase6_eval_%j.out

# Phase 6 on Wulver (no local compute): full pytest (now includes the
# metric-suite and bootstrap tests), then the evaluation orchestrator.
# It reloads the saved Stage 3 checkpoints, re-rolls every method over the
# validation segment, computes the full metric suite with moving-block
# bootstrap CIs (emphasis on the OOD fold), and produces the exposure-
# trajectory and dissociation analyses plus the recurrence ablation.
# Heavier ablation arms (reward-frontier retrains, QP vs top-k, weakened
# Layer 1, covariance choice) are separate costed runs, not launched here.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/three_layer_invar}"
VENV="${VENV:-$HOME/three_layer_invar-env}"
cd "$REPO_DIR"

module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python -m pip install \
  gymnasium==0.29.1 stable-baselines3==2.3.2 sb3-contrib==2.3.0 \
  matplotlib==3.8.4 >/dev/null

export PYTHONPATH="$REPO_DIR"
echo "Phase 6 pytest starting on $(hostname)"
python -m pytest -q

echo "Phase 6 evaluation starting"
python -u -m src.evaluation.evaluate \
  --base configs/base.yaml \
  --layer1 configs/layer1.yaml \
  --layer2 configs/layer2.yaml \
  --layer3 configs/layer3.yaml \
  --stage3 configs/stage3.yaml \
  --folds configs/folds.yaml

echo "Phase 6 complete"
