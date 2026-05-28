#!/bin/bash
#SBATCH --job-name=tli-fp-p6
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=sp500_fp_phase6_%j.out

# First-pass real-data Phase 6 evaluation, seeds 42/43/44.
# Runs after Stage 3 (Slurm afterok dependency). Produces the per-fold
# metric table, OOD bootstrap CIs, and the dissociation result on the
# real S&P 500 panel under the InVAR two-regime protocol.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/three_layer_invar}"
VENV="${VENV:-$HOME/three_layer_invar-env}"
cd "$REPO_DIR"
module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python -m pip install \
  gymnasium==0.29.1 stable-baselines3==2.3.2 sb3-contrib==2.3.0 \
  matplotlib==3.8.4 pyarrow==15.0.2 >/dev/null
export PYTHONPATH="$REPO_DIR"
python -u -m src.evaluation.evaluate \
  --base configs/base_sp500.yaml --layer1 configs/layer1.yaml \
  --layer2 configs/layer2.yaml --layer3 configs/layer3.yaml \
  --stage3 configs/stage3_sp500.yaml \
  --folds configs/folds_sp500_invar.yaml \
  --seeds 42,43,44
echo "first-pass Phase 6 complete"
