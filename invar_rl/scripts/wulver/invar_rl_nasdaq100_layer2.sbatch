#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# NASDAQ-100 Phase 4 Layer 2 mean-variance QP weight generation.
# One sbatch slot processes 5 folds for ONE seed (both protocols
# per fold per seed). Submit 5 instances with SLOT=0..4 to span
# seeds 42..46 in parallel (single-job-per-sbatch per
# [[feedback_wulver_single_job]]). Skip-if-output-exists guard on
# the per-cell summary JSON.
#
# QP work is CPU-bound (cvxpy + SCS over ~100 names per test day);
# the GPU allocation is only needed by the in-flight Layer 1 score
# regeneration path if outputs/nasdaq100/layer1/scores/fold*_seed*.parquet
# is missing for this cell (Phase 3 hit a CUDA-to-numpy bug fixed in
# commit d47860d). If scores already exist for all 25 cells, this
# script does not touch the GPU.
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_nasdaq100_layer2.sbatch
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
SEED="${SEEDS[$SLOT]}"

OUT_DIR_ROOT="outputs/nasdaq100"
LOG_DIR="invar_rl/logs/nasdaq100_layer2"
SUMMARY_DIR="${OUT_DIR_ROOT}/layer2/summary"
mkdir -p "$SUMMARY_DIR" "$LOG_DIR" \
         "${OUT_DIR_ROOT}/layer2/weights/ls" \
         "${OUT_DIR_ROOT}/layer2/weights/lo"

echo "host=$(hostname) slot=$SLOT seed=$SEED start=$(date -Is)"

for F in 1 2 3 4 5; do
  SUMMARY="${SUMMARY_DIR}/fold${F}_seed${SEED}.json"
  LS_OUT="${OUT_DIR_ROOT}/layer2/weights/ls/fold${F}_seed${SEED}.parquet"
  LO_OUT="${OUT_DIR_ROOT}/layer2/weights/lo/fold${F}_seed${SEED}.parquet"
  if [ -f "$SUMMARY" ] && [ -f "$LS_OUT" ] && [ -f "$LO_OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$F seed=$SEED: outputs exist"
    continue
  fi
  LF="${LOG_DIR}/fold${F}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] solving fold=$F seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.nasdaq100_layer2_qp \
    --fold "$F" --seed "$SEED" --protocol both \
    --output-dir-root "$OUT_DIR_ROOT" \
    > "$LF" 2>&1 || {
      echo "FAILED fold=$F seed=$SEED ; see $LF"
      continue
    }
done

echo "DONE slot=$SLOT seed=$SEED $(date -Is)"
