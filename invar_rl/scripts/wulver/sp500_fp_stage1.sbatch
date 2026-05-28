#!/bin/bash
#SBATCH --job-name=tli-fp-s1
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=sp500_fp_stage1_%j.out

# First-pass real-data Stage 1: InVAR 5-fold protocol, seeds 42/43/44,
# reduced budget (5 epochs, train-day stride 4).
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/three_layer_invar}"
VENV="${VENV:-$HOME/three_layer_invar-env}"
cd "$REPO_DIR"
module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python -m pip install pyarrow==15.0.2 >/dev/null
export PYTHONPATH="$REPO_DIR"
python -u -m src.training.stage1_rank \
  --base configs/base_sp500.yaml --layer1 configs/layer1.yaml \
  --folds configs/folds_sp500_invar.yaml \
  --seeds 42,43,44 --epochs 5 --train-day-stride 4
echo "first-pass Stage 1 complete"
