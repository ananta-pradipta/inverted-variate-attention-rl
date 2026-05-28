#!/bin/bash
#SBATCH --job-name=tli-stage3
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=stage3_rl_%j.out

# Full Stage 3 sweep (Phase 5 gated run). Trains the recurrent-PPO
# controller and the RL/non-RL baselines on the exact-replay environment
# across the seed set, with Layers 1 and 2 frozen and precomputed, and
# writes the first results table.
#
# Prerequisite: the frozen lower-stack checkpoints must already exist.
# With the default stage3.lower_stack=stage2 this means the Stage 2 sweep
# (stage2_decision.sbatch) must have COMPLETED first. Launch this only
# after that job finishes.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/three_layer_invar}"
VENV="${VENV:-$HOME/three_layer_invar-env}"
cd "$REPO_DIR"

module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python -m pip install \
  gymnasium==0.29.1 stable-baselines3==2.3.2 sb3-contrib==2.3.0 >/dev/null

export PYTHONPATH="$REPO_DIR"
echo "Stage 3 sweep starting on $(hostname)"
python -u -m src.training.stage3_rl \
  --base configs/base.yaml \
  --layer1 configs/layer1.yaml \
  --layer2 configs/layer2.yaml \
  --layer3 configs/layer3.yaml \
  --stage3 configs/stage3.yaml \
  --folds configs/folds.yaml

echo "Stage 3 sweep complete"
