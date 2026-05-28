#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# NASDAQ-100 Phase 5 Layer 3 SAC controller training. One sbatch slot
# processes 5 folds for ONE (seed, protocol) combination. Submit 10
# parallel slots (SLOT=0..9) to span 5 seeds x 2 protocols (50 cells
# total). Single-job-per-sbatch per [[feedback_wulver_single_job]];
# generic job name per [[feedback_wulver_job_names]].
#
# Slot layout (mod 5 = seed index, div 5 = protocol index):
#   SLOT 0..4 -> protocol=ls, seeds=42,43,44,45,46
#   SLOT 5..9 -> protocol=lo, seeds=42,43,44,45,46
#
# SAC training is GPU-light (twin-Q MLPs, ~hundreds of obs per second
# off the precomputed tape); the precompute step solves Layer 2 QP
# via cvxpy + SCS on CPU. ~30 min wall per cell, 5 cells per slot,
# 10 slots in parallel -> ~2.5h wall end-to-end.
#
# Skip-if-output-exists guard on the per-(fold, seed, protocol) test
# parquet AND the per-cell summary JSON.
#
# Submit (10 parallel slots):
#   for SLOT in 0 1 2 3 4 5 6 7 8 9; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_nasdaq100_layer3.sbatch
#   done

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

SLOT="${SLOT:-0}"
SEEDS=(42 43 44 45 46)
PROTOS=(ls lo)
SEED="${SEEDS[$((SLOT % 5))]}"
PROTO="${PROTOS[$((SLOT / 5))]}"
TIMESTEPS="${TIMESTEPS:-20000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"

OUT_ROOT="${OUT_ROOT:-outputs/nasdaq100/layer3}"
LOG_DIR="${LOG_DIR:-invar_rl/logs/nasdaq100_layer3}"
CKPT_ROOT="${CKPT_ROOT:-outputs/nasdaq100/layer1/_ckpt}"
mkdir -p "${OUT_ROOT}/${PROTO}/summary" "$LOG_DIR"

echo "host=$(hostname) slot=$SLOT seed=$SEED proto=$PROTO start=$(date -Is)"
nvidia-smi | head -8 || true

for F in 1 2 3 4 5; do
  CKPT="${CKPT_ROOT}/fold${F}_seed${SEED}_full.pt"
  OUT="${OUT_ROOT}/${PROTO}/fold${F}_seed${SEED}.parquet"
  SUMMARY="${OUT_ROOT}/${PROTO}/summary/fold${F}_seed${SEED}.json"
  if [ ! -f "$CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$F seed=$SEED proto=$PROTO: ckpt missing $CKPT"
    continue
  fi
  if [ -f "$OUT" ] && [ -f "$SUMMARY" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$F seed=$SEED proto=$PROTO: outputs exist"
    continue
  fi
  LF="${LOG_DIR}/fold${F}_seed${SEED}_${PROTO}.log"
  echo "[$(date +%H:%M:%S)] sac-train fold=$F seed=$SEED proto=$PROTO"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.nasdaq100_layer3_sac \
    --fold "$F" --seed "$SEED" --protocol "$PROTO" \
    --total-timesteps "$TIMESTEPS" \
    --eval-freq "$EVAL_FREQ" \
    --output-dir-root "$OUT_ROOT" \
    --layer1-ckpt-root "$CKPT_ROOT" \
    --layer3 invar_rl/configs/layer3.yaml \
    --stage3 invar_rl/configs/stage3.yaml \
    > "$LF" 2>&1 || {
      echo "FAILED fold=$F seed=$SEED proto=$PROTO ; see $LF"
      continue
    }
done

echo "DONE slot=$SLOT seed=$SEED proto=$PROTO $(date -Is)"
