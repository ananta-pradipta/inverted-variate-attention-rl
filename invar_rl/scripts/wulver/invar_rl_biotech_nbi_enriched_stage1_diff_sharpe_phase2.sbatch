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

# Option C (DSL = differentiable Sharpe surrogate at L1 finetune) Phase 2
# 25-cell sweep on biotech_nbi_enriched. Mirrors
# invar_rl_sp500_stage1_diff_sharpe_phase2.sbatch but routes to the
# NBI-enriched panel (panel_kind=biotech_nbi_enriched) via the new shared
# driver invar_rl.training.universe_layer1_diff_sharpe_eval. K=25 per side
# matches the NBI-enriched wrapper top-K cardinality.
#
# 5 slots, one fold per slot, 5 seeds (42..46) serial per slot.
# Generic job name model-eval per [[feedback_wulver_job_names]].
# Single sbatch with internal seed loop per [[feedback_wulver_single_job]].
#
# Submit (one job per fold, 5 jobs total):
#   for F in 1 2 3 4 5; do
#     sbatch --export=ALL,FOLD=$F \
#       invar_rl/scripts/wulver/invar_rl_biotech_nbi_enriched_stage1_diff_sharpe_phase2.sbatch
#   done

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Persist the full layer-1 state_dict so the layer-3 SAC eval can load it.
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
DIFF_SHARPE_WEIGHT="${DIFF_SHARPE_WEIGHT:-0.2}"
DIFF_SHARPE_TAU="${DIFF_SHARPE_TAU:-0.1}"
DIFF_SHARPE_BDAYS="${DIFF_SHARPE_BDAYS:-16}"

OUT_DIR="outputs/biotech_nbi_enriched/layer1_diff_sharpe_phase2"
LOG_DIR="invar_rl/logs/biotech_nbi_enriched_layer1_diff_sharpe_phase2"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD} K=${DIFF_SHARPE_K} loss=diff_sharpe"
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
  echo "[$(date +%H:%M:%S)] cell fold=$FOLD seed=$SEED loss=diff_sharpe K=${DIFF_SHARPE_K}"
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

# Stage 1.a (pretrain) is owned by the driver: it short-circuits when
# fold{F}_encoder.pt already exists. Series loop ensures only the
# first seed pays the pretrain cost.
for S in 42 43 44 45 46; do
  run_cell "$S"
done

echo "DONE fold=${FOLD} $(date -Is)"
