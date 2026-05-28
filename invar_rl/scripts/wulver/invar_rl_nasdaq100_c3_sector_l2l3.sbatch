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

# C3 cross-universe (2026-05-27): L2L3 SAC training on NASDAQ-100 with
# C3 sector Layer-1 ckpts. Mirrors the canonical NDX L2L3 path
# (invar_rl_nasdaq100_layer3.sbatch, QP default wrapper -> pool +1.194
# reference) but loads Layer 1 from
# outputs/nasdaq100/layer1_c3_sector/_ckpt/ instead of
# outputs/nasdaq100/layer1/_ckpt/. Same nasdaq100_layer3_sac driver,
# same Phase 5 SAC hyperparams, same 20k timesteps; only the L1 ckpts
# change.
#
# Slot layout (mod 5 = seed index, div 5 = protocol index):
#   SLOT 0..4 -> protocol=ls, seeds=42,43,44,45,46
#   SLOT 5..9 -> protocol=lo, seeds=42,43,44,45,46
#
# Submit (10 parallel slots):
#   for SLOT in 0 1 2 3 4 5 6 7 8 9; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_nasdaq100_c3_sector_l2l3.sbatch
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

OUT_ROOT="outputs/nasdaq100/layer3_c3_sector"
LOG_DIR="invar_rl/logs/nasdaq100_c3_sector_l2l3"
CKPT_ROOT="outputs/nasdaq100/layer1_c3_sector/_ckpt"
mkdir -p "${OUT_ROOT}/${PROTO}/summary" "$LOG_DIR"

echo "host=$(hostname) slot=$SLOT seed=$SEED proto=$PROTO start=$(date -Is)"
echo "ckpt_root=$CKPT_ROOT out_root=$OUT_ROOT"
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
  echo "[$(date +%H:%M:%S)] sac-train fold=$F seed=$SEED proto=$PROTO (C3 SECTOR L1 ckpt)"
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
