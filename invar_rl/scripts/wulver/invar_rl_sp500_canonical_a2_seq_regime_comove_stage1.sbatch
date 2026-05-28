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

# A2 (2026-05-27): Sequential pretrain regime -> co-movement clustering
# 25-cell SP500 sweep. Single sbatch with an internal loop over the
# 5 folds x 5 seeds per [[feedback_wulver_single_job]]; 5 parallel
# slots round-robin so each slot owns 5 of the 25 cells.
#
# Stage 1a (per fold): canonical regime-coherent InfoNCE for
# PRETRAIN_EPOCHS epochs (default 10). Selector = k-means-8 day-level
# nearest-neighbours on the 14-d standardised episode-key fingerprint.
# Byte-identical to the canonical clpretrain Stage 1.
#
# Stage 1b (per fold): per-day per-stock SupCon InfoNCE for
# COMOVE_EPOCHS epochs (default 5) with positives = same-day same-co-
# movement-cluster peers and negatives = same-day different-cluster
# peers. Cluster ids come from a per-fold 252-day rolling correlation
# k-means (K=8) on the TRAIN-segment returns matrix only (no val/test
# leakage); cached at
# cache/pretrain_improvements/comovement/sp500/foldF/cluster_ids.parquet.
# The Stage-1a encoder weights are loaded into the Stage-1b backbone
# (init_from_ckpt=True); only the encoder is carried into Stage 2.
#
# Stage 2 (per seed): canonical BANKLESS InVAR finetune with the
# canonical cs_mse loss. Byte-identical to the canonical 25-cell
# retrain except for which encoder weights Stage 2 loads.
#
# Diff vs invar_rl_sp500_canonical_c3_sector_stage1.sbatch:
#   - SLOT-based dispatch (5 slots x 5 cells); identical layout.
#   - Output dir outputs/sp500/layer1_a2_seq_regime_comove/
#   - Stage 1 PRETRAIN call swaps the sector flags for:
#       --pretrain-stages regime,comovement
#       --pretrain-comovement-universe-id sp500
#       --pretrain-comovement-n-clusters 8
#       --pretrain-comovement-window 252
#       --pretrain-comovement-epochs 5
#   - Stage 2 finetune call unchanged (canonical cs_mse).
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_sp500_canonical_a2_seq_regime_comove_stage1.sbatch
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

OUT_DIR="outputs/sp500/layer1_a2_seq_regime_comove"
LOG_DIR="invar_rl/logs/sp500_layer1_a2_seq_regime_comove"
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

echo "host=$(hostname) start=$(date -Is) slot=$SLOT total_tuples=$N_TUPLES stages=regime,comovement K=${COMOVE_K} comove_epochs=${COMOVE_EPOCHS}"
nvidia-smi | head -8 || true

# Helper: run Stage 1a + 1b pretrain for a single FOLD if the encoder
# ckpt does not exist yet. The A2 sequential pretrain is fold-keyed
# (uses seed=42 by convention; Stage 2 seeds reuse the same encoder).
ensure_pretrain() {
  local FOLD="$1"
  local ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
  if [ -f "$ENC_CKPT" ]; then
    return 0
  fi
  local PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD (A2 stages=regime,comovement K=$COMOVE_K)"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed 42 \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --pretrain_only \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --output_dir "$OUT_DIR" \
    --pretrain-stages regime,comovement \
    --pretrain-comovement-universe-id sp500 \
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
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --skip_pretrain \
    --finetune_epochs "$FINETUNE_EPOCHS" \
    --output_dir "$OUT_DIR" \
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
