#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# A1 cross-universe (2026-05-27): Stage-1 SEQUENTIAL pretrain
# regime -> sector on NASDAQ-100 (panel_kind=nasdaq100). 25-cell sweep
# mirroring the SP500 A1 layout (5 folds x 5 seeds), 5 parallel slots
# round-robin so each slot owns 5 cells.
#
# Stage 1.a (per fold, idempotent ckpt): SEQUENTIAL contrastive
# pretrain that runs TWO stages back-to-back. Stage 1a-regime uses the
# canonical k-means-8 day-level InfoNCE selector for PRETRAIN_EPOCHS
# epochs (default 10). Stage 1a-sector continues training the SAME
# encoder weights under the C3 per-stock same-sector InfoNCE selector
# for another PRETRAIN_EPOCHS epochs (default 10). Total Stage-1
# epochs = 2 * PRETRAIN_EPOCHS = 20 by default. Only the final encoder
# is carried into Stage 2.
#
# Stage 1.b (per seed): finetune of the canonical BANKLESS InVAR with
# the SAME canonical cs_mse Stage-2 loss as the headline. The only
# difference vs the canonical 25-cell retrain is which encoder weights
# Stage 2 loads.
#
# Diff vs invar_rl_sp500_canonical_a1_seq_regime_sector_stage1.sbatch:
#   - panel_kind = nasdaq100 (via the dedicated NDX runner module
#     invar_rl.training.nasdaq100_layer1_a1_seq_regime_sector; SP500
#     routes through invar_rl.training.stage1_rank).
#   - Output dir outputs/nasdaq100/layer1_a1_seq_regime_sector/
#   - --pretrain-sector-universe-id nasdaq100 (resolves
#     cache/sector_labels/nasdaq100.parquet, 11 GICS sectors).
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_nasdaq100_a1_seq_regime_sector_stage1.sbatch
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
SECTOR_UNIVERSE="${SECTOR_UNIVERSE:-nasdaq100}"

OUT_DIR="outputs/nasdaq100/layer1_a1_seq_regime_sector"
LOG_DIR="invar_rl/logs/nasdaq100_layer1_a1_seq_regime_sector"
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

echo "host=$(hostname) start=$(date -Is) slot=$SLOT total_tuples=$N_TUPLES stages=regime,sector universe=${SECTOR_UNIVERSE}"
nvidia-smi | head -8 || true

# Helper: run Stage 1.a SEQUENTIAL pretrain for a single FOLD if the
# encoder ckpt does not exist yet. The contrastive pretrain is
# fold-keyed (uses seed=42 by convention; Stage 2 seeds reuse the
# same encoder).
ensure_pretrain() {
  local FOLD="$1"
  local ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
  if [ -f "$ENC_CKPT" ]; then
    return 0
  fi
  local PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD (A1 SEQUENTIAL regime,sector, universe=${SECTOR_UNIVERSE})"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.nasdaq100_layer1_a1_seq_regime_sector \
    --fold "$FOLD" --seed 42 \
    --output-dir "$OUT_DIR" \
    --panel-end 2025-12-31 \
    --pretrain-only \
    --pretrain-epochs "$PRETRAIN_EPOCHS" \
    --pretrain-stages regime,sector \
    --pretrain-sector-universe-id "$SECTOR_UNIVERSE" \
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
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.nasdaq100_layer1_a1_seq_regime_sector \
    --fold "$FOLD" --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    --panel-end 2025-12-31 \
    --skip-pretrain \
    --finetune-epochs "$FINETUNE_EPOCHS" \
    --pretrain-sector-universe-id "$SECTOR_UNIVERSE" \
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
