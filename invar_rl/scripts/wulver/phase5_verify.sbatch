#!/bin/bash
#SBATCH --job-name=tli-phase5-verify
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=phase5_verify_%j.out

# Phase 5 code verification on Wulver (no local compute): add the RL
# dependencies to the project venv, then run the full pytest suite,
# including the Stage 3 agent-construction and non-RL-baseline tests.
# This does NOT run the Stage 3 sweep (that needs the Stage 2 frozen
# checkpoints and is the gated run in stage3_rl.sbatch).

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/three_layer_invar}"
VENV="${VENV:-$HOME/three_layer_invar-env}"
cd "$REPO_DIR"

module load python3 2>/dev/null || true
source "$VENV/bin/activate"

python -m pip install \
  gymnasium==0.29.1 stable-baselines3==2.3.2 sb3-contrib==2.3.0

export PYTHONPATH="$REPO_DIR"
echo "Phase 5 verification starting on $(hostname)"
python -m pytest -q
echo "Phase 5 verification complete"
