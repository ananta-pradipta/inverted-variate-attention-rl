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

# Option C (F2 ListMLE soft + F3 Cross-stock attention compose) Phase 2
# 25-cell sweep on SP500. 5 slots, one fold per slot, 5 seeds (42..46)
# serial per slot. Mirrors the F2 + F3 Phase 2 sbatches but:
#   - --loss-config listmle_soft (ListMLE weight 0.5, cs_mse weight 0.7)
#   - --cross-stock-attn (F3 head-boundary cross-stock attention block)
# applied jointly on the finetune call.
#
# Goal: soften the F2 listmle F2-vs-F5 trade-off while keeping the F2
# stress-fold lift, AND test whether F3's cross-stock attention composes
# additively. Phase 2 SP500 baseline pool = canonical +0.899. Pre-reg
# success: pool >= +0.95, F2 >= +0.5, F5 >= +1.5.
#
# Generic job name model-eval per [[feedback_wulver_job_names]]. One
# sbatch with internal seed loop per [[feedback_wulver_single_job]].
#
# Submit (one job per fold, 5 jobs total):
#   for F in 1 2 3 4 5; do
#     sbatch --export=ALL,FOLD=$F \
#       invar_rl/scripts/wulver/invar_rl_sp500_stage1_optionc_phase2.sbatch
#   done

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Persist the full layer-1 state_dict so the wrapper / SAC eval can
# load it. The save block writes the F3 flags into the blob so
# canonical_runner reads them back at load time.
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

OUT_DIR="outputs/sp500/layer1_optionc_phase2"
LOG_DIR="invar_rl/logs/sp500_layer1_optionc_phase2"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD} loss=listmle_soft cross_stock_attn=on"
nvidia-smi | head -8 || true

# Stage 1.a: per-fold encoder pretrain. The Stage-1 SSL objective is
# SimCLR InfoNCE and is loss_config agnostic; pretrain depends only on
# the fold, not seed or loss_config. CSA flag is recorded for cfg
# consistency but is a no-op during pretrain. Reuse the cached encoder
# ckpt if already produced.
ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
if [ -f "$ENC_CKPT" ]; then
  echo "[$(date +%H:%M:%S)] fold=$FOLD: encoder ckpt exists, skip pretrain"
else
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
    --cross-stock-attn \
    > "$PLF" 2>&1
fi

# Stage 1.b: finetune each seed (42..46) serially with both F2 soft
# ListMLE and F3 cross-stock attention. Skip cells that already have
# ckpt + JSON.
run_finetune() {
  local SEED="$1"
  local FULL_CKPT="${CKPT_DIR}/fold${FOLD}_seed${SEED}_full.pt"
  local OUT_JSON="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ -f "$FULL_CKPT" ] && [ -f "$OUT_JSON" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED (ckpt + json present)"
    return
  fi
  local LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] finetune fold=$FOLD seed=$SEED loss=listmle_soft cross_stock_attn=on"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --skip_pretrain \
    --finetune_epochs "$FINETUNE_EPOCHS" \
    --output_dir "$OUT_DIR" \
    --loss-config listmle_soft \
    --listmle-weight 0.5 \
    --cs-mse-weight 0.7 \
    --cross-stock-attn \
    > "$LF" 2>&1
}

for S in 42 43 44 45 46; do
  run_finetune "$S"
done

echo "DONE fold=${FOLD} $(date -Is)"
