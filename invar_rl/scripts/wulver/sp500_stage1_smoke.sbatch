#!/bin/bash
#SBATCH --job-name=tli-sp500-stage1
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=sp500_stage1_smoke_%j.out

# Real-data Stage 1 sub-gate: does the InVAR ranker train end-to-end on the
# real S&P 500 panel and produce positive validation rank IC across the
# three real regimes? Reduced budget (seed 42, all 3 folds, 4 epochs,
# train-day stride 5) for a fast read before wiring Stage 2/3/6.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/three_layer_invar}"
VENV="${VENV:-$HOME/three_layer_invar-env}"
cd "$REPO_DIR"

module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python -m pip install pyarrow==15.0.2 >/dev/null

export PYTHONPATH="$REPO_DIR"
echo "Real-data Stage 1 sub-gate on $(hostname)"
python -u -m src.training.stage1_rank \
  --base configs/base_sp500.yaml \
  --layer1 configs/layer1.yaml \
  --folds configs/folds_sp500_invar.yaml \
  --seeds 42 \
  --epochs 4 \
  --train-day-stride 5
echo "Real-data Stage 1 sub-gate complete"
