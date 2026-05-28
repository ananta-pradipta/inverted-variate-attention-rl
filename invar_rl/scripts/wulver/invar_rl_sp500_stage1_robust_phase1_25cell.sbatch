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

# Robust-InVAR-RL Phase 1: SP500 Stage 1 (Layer 1) retrain with
# group-DRO + top-bottom loss. 5 fold slots in parallel; each slot
# does 1 fold pretrain + 5 seeds finetune serially.
#
# Single sbatch per [[feedback_wulver_single_job]]; generic job name
# per [[feedback_wulver_job_names]]; one slot per fold so each slot
# amortises the pretrain across its 5 finetune seeds.
#
# Submit (5 parallel slots, one per fold):
#   for F in 1 2 3 4 5; do
#     sbatch --export=ALL,FOLD=$F \
#       invar_rl/scripts/wulver/invar_rl_sp500_stage1_robust_phase1_25cell.sbatch
#   done

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

export INVAR_SAVE_FULL_STATE=1

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

FOLD="${FOLD:?FOLD must be set via --export=FOLD=<1..5>}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
ETA_GDRO="${ETA_GDRO:-0.05}"
LAMBDA_TB="${LAMBDA_TB:-0.10}"
M_TB="${M_TB:-50}"

OUT_DIR="invar_rl/results/stage1_robust_phase1"
LOG_DIR="invar_rl/logs/stage1_robust_phase1"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD}"
echo "gdro: eta=${ETA_GDRO} lambda_tb=${LAMBDA_TB} M=${M_TB}"
nvidia-smi | head -8 || true

# Stage 1.a: per-fold encoder pretrain (canonical, not changed by gdro).
ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
if [ ! -f "$ENC_CKPT" ]; then
  PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed 42 \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --pretrain_only \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --output_dir "$OUT_DIR" \
    > "$PLF" 2>&1
else
  echo "[$(date +%H:%M:%S)] encoder ckpt present; skip pretrain"
fi

# Stage 1.b: gdro finetune across 5 seeds for this fold.
run_finetune() {
  local SEED="$1"
  local FULL_CKPT="${CKPT_DIR}/fold${FOLD}_seed${SEED}_full.pt"
  local OUT_JSON="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ -f "$FULL_CKPT" ] && [ -f "$OUT_JSON" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED (ckpt + json present)"
    return
  fi
  local LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] gdro finetune fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --skip_pretrain \
    --finetune_epochs "$FINETUNE_EPOCHS" \
    --output_dir "$OUT_DIR" \
    --use-group-dro \
    --eta-gdro "$ETA_GDRO" \
    --lambda-top-bottom "$LAMBDA_TB" \
    --m-top-bottom "$M_TB" \
    --gdro-universe-id sp500 \
    > "$LF" 2>&1
}

for S in 42 43 44 45 46; do
  run_finetune "$S"
done

echo "DONE fold=${FOLD} $(date -Is)"
