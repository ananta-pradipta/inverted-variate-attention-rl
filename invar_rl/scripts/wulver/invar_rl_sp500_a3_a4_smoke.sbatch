#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# A3 + A4 (2026-05-27): one-cell SP500 smoke (fold1 seed42, 2 pretrain
# epochs each) to validate the new sequential 3-stage (A3) and joint
# multi-objective (A4) Stage-1 pretrain code paths end-to-end on Wulver
# before launching the full 25-cell sweeps. Writes encoder + full ckpts
# and the Stage-2 finetune json under a throwaway _smoke output root.

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export INVAR_SAVE_FULL_STATE=1

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

FOLD=1
SEED=42
PEPOCHS=2
FEPOCHS=2

echo "host=$(hostname) start=$(date -Is) A3+A4 smoke fold=$FOLD seed=$SEED"
nvidia-smi | head -8 || true

# ---- A3: sequential regime,sector,comovement ----
A3_OUT="outputs/sp500/_smoke/layer1_a3_seq_regime_sector_comove"
mkdir -p "$A3_OUT"
echo "[$(date +%H:%M:%S)] A3 pretrain (regime,sector,comovement)"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
  --fold "$FOLD" --seed "$SEED" \
  --panel_kind lattice_native --panel_end 2025-12-31 --two_regime_val \
  --pretrain_only --pretrain_epochs "$PEPOCHS" \
  --output_dir "$A3_OUT" \
  --pretrain-stages regime,sector,comovement \
  --pretrain-sector-universe-id sp500 \
  --pretrain-comovement-universe-id sp500 \
  --pretrain-comovement-epochs 2
echo "[$(date +%H:%M:%S)] A3 finetune"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
  --fold "$FOLD" --seed "$SEED" \
  --panel_kind lattice_native --panel_end 2025-12-31 --two_regime_val \
  --skip_pretrain --finetune_epochs "$FEPOCHS" \
  --output_dir "$A3_OUT"

# ---- A4: joint regime+sector+comove ----
A4_OUT="outputs/sp500/_smoke/layer1_a4_joint"
mkdir -p "$A4_OUT"
echo "[$(date +%H:%M:%S)] A4 pretrain (joint)"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
  --fold "$FOLD" --seed "$SEED" \
  --panel_kind lattice_native --panel_end 2025-12-31 --two_regime_val \
  --pretrain_only --pretrain_epochs "$PEPOCHS" \
  --output_dir "$A4_OUT" \
  --pretrain-positive-method joint \
  --pretrain-sector-universe-id sp500 \
  --pretrain-comovement-universe-id sp500 \
  --pretrain-joint-weight-regime 1.0 \
  --pretrain-joint-weight-sector 1.0 \
  --pretrain-joint-weight-comove 1.0
echo "[$(date +%H:%M:%S)] A4 finetune"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
  --fold "$FOLD" --seed "$SEED" \
  --panel_kind lattice_native --panel_end 2025-12-31 --two_regime_val \
  --skip_pretrain --finetune_epochs "$FEPOCHS" \
  --output_dir "$A4_OUT"

echo "DONE $(date -Is)"
echo "=== A3 outputs ==="; ls -la "$A3_OUT" "$A3_OUT/_ckpt" 2>/dev/null || true
echo "=== A4 outputs ==="; ls -la "$A4_OUT" "$A4_OUT/_ckpt" 2>/dev/null || true
