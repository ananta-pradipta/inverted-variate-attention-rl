#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# DSL Phase 2 tune B: NBI-enriched with weight=0.5 (tau unchanged 0.1).
# Hypothesis: higher Sharpe-surrogate weight may further amplify the
# F4 +1.293 and F5 +0.373 weak-fold recovery observed under the
# canonical DSL preset (weight=0.2), at potential cost to F1/F2.
#
# Reuses universe_layer1_diff_sharpe_eval driver byte-for-byte; only
# overrides DIFF_SHARPE_WEIGHT default. K=25 unchanged (NBI wrapper
# top-K). tau=0.1 unchanged so the soft-topK sharpness is held fixed.
#
# Single sbatch with internal seed loop per single-job feedback rule.
# Generic job name model-eval per job-name feedback rule.
#
# Submit (one job per fold, 5 jobs total):
#   for F in 1 2 3 4 5; do
#     sbatch --export=ALL,FOLD=$F \
#       invar_rl/scripts/wulver/invar_rl_biotech_nbi_enriched_diff_sharpe_tuneB_stage1.sbatch
#   done

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

FOLD="${FOLD:?FOLD must be set via --export=FOLD=<1..5>}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
DIFF_SHARPE_K="${DIFF_SHARPE_K:-25}"
DIFF_SHARPE_WEIGHT="${DIFF_SHARPE_WEIGHT:-0.5}"
DIFF_SHARPE_TAU="${DIFF_SHARPE_TAU:-0.1}"
DIFF_SHARPE_BDAYS="${DIFF_SHARPE_BDAYS:-16}"

OUT_DIR="outputs/biotech_nbi_enriched/layer1/diff_sharpe_phase2_tuneB"
LOG_DIR="invar_rl/logs/biotech_nbi_enriched_layer1_diff_sharpe_phase2_tuneB"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD} K=${DIFF_SHARPE_K} tau=${DIFF_SHARPE_TAU} weight=${DIFF_SHARPE_WEIGHT} loss=diff_sharpe"
nvidia-smi | head -8 || true

run_cell() {
  local SEED="$1"
  local FULL_CKPT="${CKPT_DIR}/fold${FOLD}_seed${SEED}_full.pt"
  local OUT_JSON="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ -f "$FULL_CKPT" ] && [ -f "$OUT_JSON" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED (ckpt + json present)"
    return
  fi
  local LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] cell fold=$FOLD seed=$SEED loss=diff_sharpe K=${DIFF_SHARPE_K} tau=${DIFF_SHARPE_TAU} w=${DIFF_SHARPE_WEIGHT}"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.universe_layer1_diff_sharpe_eval \
    --universe biotech_nbi_enriched \
    --fold "$FOLD" --seed "$SEED" \
    --output-dir-root "$OUT_DIR" \
    --pretrain-epochs "$PRETRAIN_EPOCHS" \
    --finetune-epochs "$FINETUNE_EPOCHS" \
    --panel-end 2025-12-31 \
    --diff-sharpe-weight "$DIFF_SHARPE_WEIGHT" \
    --diff-sharpe-k "$DIFF_SHARPE_K" \
    --diff-sharpe-temperature "$DIFF_SHARPE_TAU" \
    --diff-sharpe-batch-days "$DIFF_SHARPE_BDAYS" \
    > "$LF" 2>&1
}

for S in 42 43 44 45 46; do
  run_cell "$S"
done

echo "DONE fold=${FOLD} $(date -Is)"
