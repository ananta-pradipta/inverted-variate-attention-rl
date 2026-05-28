#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Option B Phase 0 smoke: 1-epoch multitask pretrain (fold=2 only) +
# 1-epoch SP500 finetune (fold=2 seed=42) + 1 SP500 L2L3 cell. Verifies
# the full pipeline end-to-end before the 5-fold pretrain + 25-cell
# finetune + 25-cell L2L3 production runs.
#
# Smoke skips the L2L3 stage by default (set RUN_L2L3=1 to include).
#
# Submit:
#   sbatch invar_rl/scripts/wulver/multitask_l1_smoke.sbatch

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export INVAR_SAVE_FULL_STATE=1

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

FOLD="${FOLD:-2}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-1}"
RUN_L2L3="${RUN_L2L3:-0}"

PRETRAIN_DIR="invar_rl/results/multitask_l1_smoke"
OUT_DIR="${PRETRAIN_DIR}/lattice_native"
LOG_DIR="invar_rl/logs/multitask_l1_smoke"
mkdir -p "$PRETRAIN_DIR" "$OUT_DIR/_ckpt" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=$FOLD seed=$SEED epochs=$EPOCHS"
nvidia-smi | head -8 || true

# Stage 1: smoke pretrain (1 epoch each stage, 3 universes, 1 fold).
echo "[$(date +%H:%M:%S)] smoke pretrain fold=$FOLD"
PYTHONPATH=$PWD python3 -u -m src.baselines.train_multitask_pretrain \
  --fold "$FOLD" --seed 42 \
  --panel_end 2025-12-31 \
  --two_regime_val \
  --pretrain_epochs "$EPOCHS" \
  --output_dir "$PRETRAIN_DIR" \
  > "${LOG_DIR}/pretrain_fold${FOLD}.log" 2>&1 || {
    echo "PRETRAIN FAILED; see ${LOG_DIR}/pretrain_fold${FOLD}.log"
    tail -60 "${LOG_DIR}/pretrain_fold${FOLD}.log"
    exit 1
  }

# Stage 2: smoke finetune (SP500 only, fold=FOLD seed=SEED, EPOCHS epochs).
echo "[$(date +%H:%M:%S)] smoke finetune fold=$FOLD seed=$SEED"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.multitask_l1_finetune \
  --universe lattice_native \
  --fold "$FOLD" --seed "$SEED" \
  --multitask-pretrain-dir "$PRETRAIN_DIR" \
  --output-dir "$OUT_DIR" \
  --finetune-epochs "$EPOCHS" \
  --panel_end 2025-12-31 \
  --two_regime_val \
  > "${LOG_DIR}/finetune_fold${FOLD}_seed${SEED}.log" 2>&1 || {
    echo "FINETUNE FAILED; see ${LOG_DIR}/finetune_fold${FOLD}_seed${SEED}.log"
    tail -60 "${LOG_DIR}/finetune_fold${FOLD}_seed${SEED}.log"
    exit 1
  }

if [ "$RUN_L2L3" = "1" ]; then
  CKPT="${OUT_DIR}/_ckpt/fold${FOLD}_seed${SEED}_full.pt"
  OUT_L2L3="outputs/sp500/stage3_rl_ablation/multitask_l1_smoke"
  TAPE_L2L3="${OUT_L2L3}/daily_tape"
  mkdir -p "$OUT_L2L3" "$TAPE_L2L3"
  echo "[$(date +%H:%M:%S)] smoke l2l3 fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage3_rl_ablation \
    --ablation equal_l2 \
    --fold "$FOLD" --seed "$SEED" \
    --layer1-ckpt "$CKPT" \
    --output-dir "$OUT_L2L3" \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --equal-topk-k 50 \
    --methods sac \
    --save-daily-tape \
    --daily-tape-dir "$TAPE_L2L3" \
    > "${LOG_DIR}/l2l3_fold${FOLD}_seed${SEED}.log" 2>&1 || {
      echo "L2L3 FAILED; see ${LOG_DIR}/l2l3_fold${FOLD}_seed${SEED}.log"
      tail -60 "${LOG_DIR}/l2l3_fold${FOLD}_seed${SEED}.log"
      exit 1
    }
fi

echo "DONE $(date -Is)"
ls -la "$PRETRAIN_DIR/_ckpt_per_universe/" 2>/dev/null || true
ls -la "$OUT_DIR/_ckpt/" 2>/dev/null || true
