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

# B1 (2026-05-27): Stage-1 HMM regime posterior contrastive selector
# 25-cell SP500 sweep. Single sbatch with an internal loop over the
# 5 folds x 5 seeds per [[feedback_wulver_single_job]]; 5 parallel
# slots round-robin so each slot owns 5 of the 25 cells (matches the
# canonical retrain layout).
#
# Stage 1.a: per-fold contrastive pretrain with the B1 HMM regime
# selector ON (n_states=4, threshold=0.7). The Stage-1 SSL objective
# stays SupCon InfoNCE; only the POSITIVE-MASK SELECTOR changes from
# L2 nearest-neighbour on the 14-d standardised episode-key fingerprint
# to cosine similarity over Gaussian HMM posteriors.
# Only the encoder is carried into Stage 2.
#
# Stage 1.b: per-seed finetune of the canonical BANKLESS InVAR with
# the SAME canonical cs_mse Stage-2 loss as the headline. The only
# difference vs the canonical 25-cell retrain is which encoder weights
# Stage 2 loads.
#
# Diff vs invar_rl_sp500_canonical_b2_pretrain_aux_stage1.sbatch:
#   - SLOT-based dispatch (5 slots x 5 cells); identical layout.
#   - Output dir outputs/sp500/layer1_b1_hmm/
#   - Stage 1 PRETRAIN call swaps the aux-head flags for:
#       --pretrain-regime-method hmm
#       --pretrain-hmm-n-states 4
#       --pretrain-hmm-positive-threshold 0.7
#       --pretrain-hmm-universe-id sp500
#   - Stage 2 finetune call unchanged (canonical cs_mse).
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_sp500_canonical_b1_hmm_stage1.sbatch
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
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
HMM_N_STATES="${HMM_N_STATES:-4}"
HMM_THRESH="${HMM_THRESH:-0.7}"

OUT_DIR="outputs/sp500/layer1_b1_hmm"
LOG_DIR="invar_rl/logs/sp500_layer1_b1_hmm"
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

echo "host=$(hostname) start=$(date -Is) slot=$SLOT total_tuples=$N_TUPLES n_states=$HMM_N_STATES thresh=$HMM_THRESH"
nvidia-smi | head -8 || true

# Helper: run Stage 1.a pretrain for a single FOLD if the encoder ckpt
# does not exist yet. The contrastive + B1 HMM pretrain is fold-keyed
# (uses seed=42 by convention; Stage 2 seeds reuse the same encoder).
ensure_pretrain() {
  local FOLD="$1"
  local ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
  if [ -f "$ENC_CKPT" ]; then
    return 0
  fi
  local PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD (B1 HMM ON, n_states=$HMM_N_STATES, thresh=$HMM_THRESH)"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed 42 \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --pretrain_only \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --output_dir "$OUT_DIR" \
    --pretrain-regime-method hmm \
    --pretrain-hmm-n-states "$HMM_N_STATES" \
    --pretrain-hmm-positive-threshold "$HMM_THRESH" \
    --pretrain-hmm-universe-id sp500 \
    > "$PLF" 2>&1
}

# Helper: run Stage 1.b finetune for a single (FOLD, SEED). The
# canonical bankless cs_mse path is preserved byte-identically; only
# the encoder weights loaded into model.temporal_encoder differ.
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

# Track which folds this slot has already pretrained on (each slot may
# need to pretrain >1 fold since round-robin spreads cells across slots,
# but pretrain is idempotent so a re-call just no-ops via the ckpt check).
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
