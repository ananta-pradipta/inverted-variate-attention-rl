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

# A2 cross-universe (2026-05-27): Stage-1 SEQUENTIAL pretrain
# regime -> co-movement on NASDAQ-100 (panel_kind=nasdaq100). 25-cell
# sweep mirroring the SP500 A2 layout (5 folds x 5 seeds), 5 parallel
# slots round-robin so each slot owns 5 cells.
#
# Stage 1.a: per-fold contrastive pretrain with the canonical regime
# k-means-8 selector (10 epochs by default; byte-identical to canonical).
# Stage 1.b: per-fold per-day per-stock SupCon InfoNCE whose positives
# are same-day same-co-movement-cluster peers. Cluster ids come from a
# per-fold 252-day rolling correlation k-means (K=8) on the TRAIN-segment
# returns matrix only; cached at
# cache/pretrain_improvements/comovement/nasdaq100/foldF/cluster_ids.parquet
# (pre-built via invar_rl.scripts.build_comovement_clusters before
# submit). Encoder ckpt persisted at
# outputs/nasdaq100/layer1_a2_seq_regime_comove/_ckpt/fold{F}_encoder.pt
# and shared across the 5 finetune seeds.
#
# Stage 2: canonical BANKLESS bankless cs_mse finetune on the NDX panel.
# The only difference vs the canonical NDX layer1 retrain is which
# encoder weights Stage 2 loads.
#
# Diff vs invar_rl_sp500_canonical_a2_seq_regime_comove_stage1.sbatch:
#   - panel_kind = nasdaq100 (via the dedicated NDX runner module
#     invar_rl.training.nasdaq100_layer1_a2_seq_regime_comove; SP500
#     routes through invar_rl.training.stage1_rank).
#   - Output dir outputs/nasdaq100/layer1_a2_seq_regime_comove/
#   - Cluster cache cache/pretrain_improvements/comovement/nasdaq100/
#   - --pretrain-comovement-universe-id nasdaq100
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_nasdaq100_a2_seq_regime_comove_stage1.sbatch
#   done

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Persist the full layer-1 state_dict so the wrapper / SAC eval can load it.
export INVAR_SAVE_FULL_STATE=1

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

SLOT="${SLOT:?SLOT must be set via --export=SLOT=0..4}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
COMOVE_EPOCHS="${COMOVE_EPOCHS:-5}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
COMOVE_K="${COMOVE_K:-8}"
COMOVE_WINDOW="${COMOVE_WINDOW:-252}"
COMOVE_UNIVERSE="${COMOVE_UNIVERSE:-nasdaq100}"

OUT_DIR="outputs/nasdaq100/layer1_a2_seq_regime_comove"
LOG_DIR="invar_rl/logs/nasdaq100_layer1_a2_seq_regime_comove"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

# 25 (fold, seed) tuples; round-robin over 5 slots so each slot owns 5.
TUPLES=()
for FOLD in 1 2 3 4 5; do
  for SEED in 42 43 44 45 46; do
    TUPLES+=("${FOLD}|${SEED}")
  done
done
N_TUPLES=${#TUPLES[@]}
N_SLOTS=5

echo "host=$(hostname) start=$(date -Is) slot=$SLOT total_tuples=$N_TUPLES stages=regime,comovement K=${COMOVE_K} comove_epochs=${COMOVE_EPOCHS} universe=${COMOVE_UNIVERSE}"
nvidia-smi | head -8 || true

# Helper: run Stage 1.a + 1.b pretrain for a single FOLD if the encoder
# ckpt does not exist yet. The A2 sequential pretrain is fold-keyed
# (uses seed=42 by convention; Stage 2 seeds reuse the same encoder).
ensure_pretrain() {
  local FOLD="$1"
  local ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
  if [ -f "$ENC_CKPT" ]; then
    return 0
  fi
  local PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD (A2 stages=regime,comovement K=$COMOVE_K universe=$COMOVE_UNIVERSE)"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.nasdaq100_layer1_a2_seq_regime_comove \
    --fold "$FOLD" --seed 42 \
    --output-dir "$OUT_DIR" \
    --panel-end 2025-12-31 \
    --pretrain-only \
    --pretrain-epochs "$PRETRAIN_EPOCHS" \
    --pretrain-stages regime,comovement \
    --pretrain-comovement-universe-id "$COMOVE_UNIVERSE" \
    --pretrain-comovement-n-clusters "$COMOVE_K" \
    --pretrain-comovement-window "$COMOVE_WINDOW" \
    --pretrain-comovement-epochs "$COMOVE_EPOCHS" \
    > "$PLF" 2>&1
}

# Helper: run Stage 2 finetune for a single (FOLD, SEED). The canonical
# bankless cs_mse path is preserved byte-identically; only the encoder
# weights loaded into model.temporal_encoder differ.
run_finetune() {
  local FOLD="$1"
  local SEED="$2"
  local OUT_JSON="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  local FULL_CKPT="${CKPT_DIR}/fold${FOLD}_seed${SEED}_full.pt"
  if [ -f "$OUT_JSON" ] && [ -f "$FULL_CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip finetune fold=$FOLD seed=$SEED (json+ckpt present)"
    return 0
  fi
  local LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] finetune fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.nasdaq100_layer1_a2_seq_regime_comove \
    --fold "$FOLD" --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    --panel-end 2025-12-31 \
    --skip-pretrain \
    --finetune-epochs "$FINETUNE_EPOCHS" \
    --pretrain-comovement-universe-id "$COMOVE_UNIVERSE" \
    --pretrain-comovement-n-clusters "$COMOVE_K" \
    --pretrain-comovement-window "$COMOVE_WINDOW" \
    --pretrain-comovement-epochs "$COMOVE_EPOCHS" \
    > "$LF" 2>&1
}

# Round-robin dispatch; ensure_pretrain is idempotent via the ckpt check
# so re-callers for the same fold no-op cheaply.
for i in $(seq 0 $((N_TUPLES - 1))); do
  if [ $((i % N_SLOTS)) -ne "$SLOT" ]; then
    continue
  fi
  IFS='|' read -r FOLD SEED <<< "${TUPLES[$i]}"
  ensure_pretrain "$FOLD" || {
    echo "FAILED pretrain fold=$FOLD; see ${LOG_DIR}/fold${FOLD}_pretrain.log"
    tail -50 "${LOG_DIR}/fold${FOLD}_pretrain.log" 2>/dev/null || true
    continue
  }
  run_finetune "$FOLD" "$SEED" || {
    echo "FAILED finetune fold=$FOLD seed=$SEED; see ${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
    tail -50 "${LOG_DIR}/fold${FOLD}_seed${SEED}.log" 2>/dev/null || true
    continue
  }
done

echo "DONE slot=${SLOT} $(date -Is)"
ls -la "$OUT_DIR/" 2>/dev/null || true
