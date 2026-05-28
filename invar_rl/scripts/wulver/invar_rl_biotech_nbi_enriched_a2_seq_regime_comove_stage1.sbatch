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
# regime -> co-movement on biotech NBI-enriched
# (panel_kind=biotech_nbi_enriched). 25-cell sweep mirroring the SP500
# A2 layout (5 folds x 5 seeds), 5 parallel slots round-robin so each
# slot owns 5 cells.
#
# Why A2 is universe-agnostic for NBI:
#   C3 sector positives needed a sub-industry parquet to avoid the
#   degenerate top-level GICS "Health Care" cohort. A2 co-movement is
#   data-driven from the train-fold returns matrix; no external
#   sub-industry mapping required. The per-fold cluster cache is
#   pre-built via
#   `invar_rl.scripts.build_comovement_clusters
#       --panel-kind biotech_nbi_enriched
#       --universe biotech_nbi_enriched --folds 1 2 3 4 5`.
#
# Stage 1.a: per-fold canonical regime k-means-8 InfoNCE (10 epochs,
# byte-identical to canonical NBI pretrain).
# Stage 1.b: per-fold per-day per-stock SupCon InfoNCE whose positives
# are same-day same-co-movement-cluster peers. Cluster cache at
# cache/pretrain_improvements/comovement/biotech_nbi_enriched/foldF/
# cluster_ids.parquet. Encoder ckpt persisted at
# outputs/biotech_nbi_enriched/layer1_a2_seq_regime_comove/_ckpt/fold{F}_encoder.pt
# and shared across the 5 finetune seeds.
#
# Stage 2: canonical BANKLESS bankless cs_mse finetune on the NBI panel.
# The only difference vs the canonical NBI layer1 retrain is which
# encoder weights Stage 2 loads.
#
# Diff vs invar_rl_nasdaq100_a2_seq_regime_comove_stage1.sbatch:
#   - panel_kind = biotech_nbi_enriched (via dedicated NBI runner module
#     invar_rl.training.biotech_nbi_enriched_layer1_a2_seq_regime_comove).
#   - Output dir outputs/biotech_nbi_enriched/layer1_a2_seq_regime_comove/
#   - Cluster cache cache/pretrain_improvements/comovement/biotech_nbi_enriched/
#   - --pretrain-comovement-universe-id biotech_nbi_enriched
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_biotech_nbi_enriched_a2_seq_regime_comove_stage1.sbatch
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
COMOVE_UNIVERSE="${COMOVE_UNIVERSE:-biotech_nbi_enriched}"

OUT_DIR="outputs/biotech_nbi_enriched/layer1_a2_seq_regime_comove"
LOG_DIR="invar_rl/logs/biotech_nbi_enriched_layer1_a2_seq_regime_comove"
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

echo "host=$(hostname) start=$(date -Is) slot=$SLOT total_tuples=$N_TUPLES stages=regime,comovement K=${COMOVE_K} comove_epochs=${COMOVE_EPOCHS} universe=${COMOVE_UNIVERSE}"
nvidia-smi | head -8 || true

ensure_pretrain() {
  local FOLD="$1"
  local ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
  if [ -f "$ENC_CKPT" ]; then
    return 0
  fi
  local PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD (A2 stages=regime,comovement K=$COMOVE_K universe=$COMOVE_UNIVERSE)"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.biotech_nbi_enriched_layer1_a2_seq_regime_comove \
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
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.biotech_nbi_enriched_layer1_a2_seq_regime_comove \
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
