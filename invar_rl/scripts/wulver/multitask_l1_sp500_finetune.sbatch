#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Option B Stage-2: SP500 (lattice_native) per-(fold, seed) finetune
# starting from the multitask-pretrained backbone. Single sbatch slot
# per [[feedback_wulver_single_job]] handling a round-robin subset of
# the 25 cells. Submit 5 instances with SLOT=0..4 for parallelism.
#
# Depends on multitask_l1_pretrain.sbatch having written
# {PRETRAIN_DIR}/_ckpt_per_universe/lattice_native/foldF_encoder.pt for
# every fold.
#
# Generic job name per [[feedback_wulver_job_names]].
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/multitask_l1_sp500_finetune.sbatch
#   done

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

SLOT="${SLOT:?SLOT must be set via --export=SLOT=0..4}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"

PRETRAIN_DIR="invar_rl/results/multitask_l1"
OUT_DIR="invar_rl/results/multitask_l1/lattice_native"
LOG_DIR="invar_rl/logs/multitask_l1_sp500_finetune"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

# Persist the full layer-1 state_dict so downstream L2L3 SAC can load it.
export INVAR_SAVE_FULL_STATE=1

# 25 (fold, seed) tuples; round-robin over 5 slots so each slot owns 5 cells.
TUPLES=()
for FOLD in 1 2 3 4 5; do
  for SEED in 42 43 44 45 46; do
    TUPLES+=("${FOLD}|${SEED}")
  done
done
N_TUPLES=${#TUPLES[@]}
N_SLOTS=5

echo "host=$(hostname) start=$(date -Is) slot=$SLOT total_tuples=$N_TUPLES"
nvidia-smi | head -8 || true

for i in $(seq 0 $((N_TUPLES - 1))); do
  if [ $((i % N_SLOTS)) -ne "$SLOT" ]; then
    continue
  fi
  IFS='|' read -r FOLD SEED <<< "${TUPLES[$i]}"
  OUT_JSON="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  FULL_CKPT="${CKPT_DIR}/fold${FOLD}_seed${SEED}_full.pt"
  if [ -f "$OUT_JSON" ] && [ -f "$FULL_CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED (json + full ckpt present)"
    continue
  fi
  MT_CKPT="${PRETRAIN_DIR}/_ckpt_per_universe/lattice_native/fold${FOLD}_encoder.pt"
  if [ ! -f "$MT_CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED: multitask ckpt missing $MT_CKPT"
    continue
  fi
  LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] finetune fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.multitask_l1_finetune \
    --universe lattice_native \
    --fold "$FOLD" --seed "$SEED" \
    --multitask-pretrain-dir "$PRETRAIN_DIR" \
    --output-dir "$OUT_DIR" \
    --finetune-epochs "$FINETUNE_EPOCHS" \
    --panel_end 2025-12-31 \
    --two_regime_val \
    > "$LF" 2>&1 || {
      echo "FAILED fold=$FOLD seed=$SEED; see $LF"
      tail -50 "$LF"
      continue
    }
done

echo "DONE slot=${SLOT} $(date -Is)"
ls -la "$OUT_DIR/" 2>/dev/null | head -30 || true
