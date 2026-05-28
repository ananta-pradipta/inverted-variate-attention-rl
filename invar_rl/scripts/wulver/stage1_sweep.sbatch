#!/bin/bash
#SBATCH --job-name=stage1-sweep
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=stage1_sweep_%j.out

# Full Stage 1 sweep for Layer 1 (the InVAR ranker).
# One job, one internal loop: src.training.stage1_rank.run iterates the full
# seed set (base.seeds) across every configured fold at the config schedule
# (no epoch or stride overrides). This is not a Slurm array; the sweep loop
# lives inside the Python entry point.
#
# Adjust REPO_DIR, the environment activation block, and --time to the course
# QOS walltime limit before submitting.

set -euo pipefail

# Project root on Wulver (the three_layer_invar directory). Override by
# exporting REPO_DIR before calling sbatch.
REPO_DIR="${REPO_DIR:-$HOME/three_layer_invar}"
cd "$REPO_DIR"

# Environment activation, matching the established Wulver job pattern.
module load python3 2>/dev/null || true
source "$HOME/phd-research-gpu-env/bin/activate"

export PYTHONPATH="$REPO_DIR"

echo "Stage 1 sweep starting in $REPO_DIR on $(hostname)"
python -u -m src.training.stage1_rank \
  --base configs/base.yaml \
  --layer1 configs/layer1.yaml \
  --folds configs/folds.yaml \
  --device cuda

echo "Stage 1 sweep complete"
