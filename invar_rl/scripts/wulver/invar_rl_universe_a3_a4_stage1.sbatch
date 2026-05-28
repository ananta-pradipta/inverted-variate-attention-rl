#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# A3 / A4 (2026-05-27): generic per-universe Stage-1 pretrain sweep
# (25 cells) driven by env vars. Used for the NDX-100 and biotech-NBI
# cross-universe transfer of the new sequential 3-stage (A3) and joint
# multi-objective (A4) pretrain. SP500 uses the dedicated stage1_rank
# sbatches; this generic one routes through
# invar_rl.training.universe_layer1_a3_a4 for non-SP500 panels.
#
# Single sbatch with internal loop per [[feedback_wulver_single_job]];
# 5 parallel slots round-robin over 25 cells. Generic job name per
# [[feedback_wulver_job_names]].
#
# Required env:
#   MODE          a3 | a4
#   PANEL_KIND    nasdaq100 | biotech_nbi_enriched
#   UNIVERSE_ID   nasdaq100 | biotech_nbi_enriched (sector + comove cache key)
#   OUT_DIR       output root (also holds _ckpt)
# Optional env:
#   PRETRAIN_EPOCHS (10), COMOVE_EPOCHS (5), FINETUNE_EPOCHS (10)
#
# Submit (5 parallel slots), e.g. A3 on NDX:
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT,MODE=a3,PANEL_KIND=nasdaq100,\
# UNIVERSE_ID=nasdaq100,\
# OUT_DIR=outputs/nasdaq100/layer1_a3_seq_regime_sector_comove \
#       invar_rl/scripts/wulver/invar_rl_universe_a3_a4_stage1.sbatch
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

SLOT="${SLOT:?SLOT must be set via --export=SLOT=0..4}"
MODE="${MODE:?MODE must be a3 or a4}"
PANEL_KIND="${PANEL_KIND:?PANEL_KIND must be set}"
UNIVERSE_ID="${UNIVERSE_ID:?UNIVERSE_ID must be set}"
OUT_DIR="${OUT_DIR:?OUT_DIR must be set}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
COMOVE_EPOCHS="${COMOVE_EPOCHS:-5}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"

LOG_DIR="invar_rl/logs/$(basename "$OUT_DIR")"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

TUPLES=()
for FOLD in 1 2 3 4 5; do
  for SEED in 42 43 44 45 46; do
    TUPLES+=("${FOLD}|${SEED}")
  done
done
N_TUPLES=${#TUPLES[@]}
N_SLOTS=5

echo "host=$(hostname) start=$(date -Is) slot=$SLOT mode=$MODE panel=$PANEL_KIND universe=$UNIVERSE_ID out=$OUT_DIR"
nvidia-smi | head -8 || true

ensure_pretrain() {
  local FOLD="$1"
  local ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
  if [ -f "$ENC_CKPT" ]; then
    return 0
  fi
  local PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD mode=$MODE panel=$PANEL_KIND"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.universe_layer1_a3_a4 \
    --mode "$MODE" \
    --panel-kind "$PANEL_KIND" \
    --fold "$FOLD" --seed 42 \
    --output-dir "$OUT_DIR" \
    --sector-universe-id "$UNIVERSE_ID" \
    --comovement-universe-id "$UNIVERSE_ID" \
    --comovement-epochs "$COMOVE_EPOCHS" \
    --pretrain-epochs "$PRETRAIN_EPOCHS" \
    --pretrain-only \
    > "$PLF" 2>&1
}

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
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.universe_layer1_a3_a4 \
    --mode "$MODE" \
    --panel-kind "$PANEL_KIND" \
    --fold "$FOLD" --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    --sector-universe-id "$UNIVERSE_ID" \
    --comovement-universe-id "$UNIVERSE_ID" \
    --comovement-epochs "$COMOVE_EPOCHS" \
    --finetune-epochs "$FINETUNE_EPOCHS" \
    --skip-pretrain \
    > "$LF" 2>&1
}

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
