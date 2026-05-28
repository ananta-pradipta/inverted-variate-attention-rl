#!/bin/bash
#SBATCH --job-name=tli-fp-s2
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=sp500_fp_stage2_%j.out

# First-pass real-data Stage 2 (decision-focused), seeds 42/43/44.
# Runs after Stage 1 (Slurm afterok dependency).
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/three_layer_invar}"
VENV="${VENV:-$HOME/three_layer_invar-env}"
cd "$REPO_DIR"
module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python -m pip install pyarrow==15.0.2 >/dev/null
export PYTHONPATH="$REPO_DIR"
python -u -m src.training.stage2_decision \
  --base configs/base_sp500.yaml --layer1 configs/layer1.yaml \
  --layer2 configs/layer2.yaml --stage2 configs/stage2.yaml \
  --folds configs/folds_sp500_invar.yaml \
  --seeds 42,43,44
echo "first-pass Stage 2 complete"
