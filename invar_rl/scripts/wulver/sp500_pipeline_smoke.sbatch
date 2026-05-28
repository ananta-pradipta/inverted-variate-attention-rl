#!/bin/bash
#SBATCH --job-name=tli-sp500-pipe
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sp500_pipeline_smoke_%j.out

# Real-data pipeline wiring smoke: chain Stage 1 -> Stage 2 -> Stage 3 ->
# Phase 6 on the InVAR 5-fold protocol, restricted to seed 42 and one fold
# (fold_1_covid) with tiny budgets and a high precompute stride. The point
# is to exercise the sp500 two-regime path through Stage 2/3/6 end to end
# and catch any wiring breakage cheaply before the full first-pass run.
# This is NOT a result.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/three_layer_invar}"
VENV="${VENV:-$HOME/three_layer_invar-env}"
cd "$REPO_DIR"

module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python -m pip install pyarrow==15.0.2 >/dev/null

export PYTHONPATH="$REPO_DIR"
F="fold_1_covid"

echo "[1/4] Stage 1 (real data, $F, seed 42)"
python -u -m src.training.stage1_rank \
  --base configs/base_sp500.yaml --layer1 configs/layer1.yaml \
  --folds configs/folds_sp500_invar.yaml \
  --seeds 42 --fold-names "$F" --epochs 3 --train-day-stride 8

echo "[2/4] Stage 2 (decision-focused, $F, seed 42)"
python -u -m src.training.stage2_decision \
  --base configs/base_sp500.yaml --layer1 configs/layer1.yaml \
  --layer2 configs/layer2.yaml --stage2 configs/stage2.yaml \
  --folds configs/folds_sp500_invar.yaml \
  --seeds 42 --fold-names "$F"

echo "[3/4] Stage 3 (RL controller + baselines, $F, seed 42)"
python -u -m src.training.stage3_rl \
  --base configs/base_sp500.yaml --layer1 configs/layer1.yaml \
  --layer2 configs/layer2.yaml --layer3 configs/layer3.yaml \
  --stage3 configs/stage3_sp500_smoke.yaml \
  --folds configs/folds_sp500_invar.yaml \
  --seeds 42 --fold-names "$F"

echo "[4/4] Phase 6 evaluation ($F, seed 42)"
python -u -m src.evaluation.evaluate \
  --base configs/base_sp500.yaml --layer1 configs/layer1.yaml \
  --layer2 configs/layer2.yaml --layer3 configs/layer3.yaml \
  --stage3 configs/stage3_sp500_smoke.yaml \
  --folds configs/folds_sp500_invar.yaml \
  --seeds 42 --fold-names "$F"

echo "Real-data pipeline smoke complete"
