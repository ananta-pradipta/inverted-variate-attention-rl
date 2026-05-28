#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Phase 2 of fairness audit issue #5 fix: retrain StockFormer whole-stack
# RL on the biotech NBI ENRICHED 22-feature panel.
# Grid: 5 folds x 5 seeds = 25 cells across 5 throttled slots.
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT,N_SLOTS=5 \
#       invar_rl/scripts/wulver/invar_rl_biotech_nbi_enriched_stockformer.sbatch
#   done

set -uo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

SLOT="${SLOT:-0}"
N_SLOTS="${N_SLOTS:-5}"

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

OUT_ROOT="outputs/biotech_nbi_enriched/whole_stack_rl/stockformer"
RAW_ROOT="${OUT_ROOT}/_raw"
LOG_DIR="invar_rl/logs/biotech_nbi_enriched_stockformer"
mkdir -p "$OUT_ROOT" "$RAW_ROOT" "$LOG_DIR"

# 5 folds x 5 seeds = 25
declare -a CELLS
IDX=0
for F in 1 2 3 4 5; do
  for SEED in 42 43 44 45 46; do
    CELLS[$IDX]="$F $SEED"
    IDX=$((IDX + 1))
  done
done

echo "host=$(hostname) slot=$SLOT/$N_SLOTS total_cells=${#CELLS[@]} start=$(date -Is)"
nvidia-smi | head -8 || true

for i in "${!CELLS[@]}"; do
  if [ $((i % N_SLOTS)) -ne "$SLOT" ]; then continue; fi
  read FOLD SEED <<< "${CELLS[$i]}"
  OUT="$OUT_ROOT/fold${FOLD}_seed${SEED}.json"
  LF="$LOG_DIR/sf_fold${FOLD}_seed${SEED}.log"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] SKIP slot=$SLOT idx=$i $OUT exists"
    continue
  fi
  echo "[$(date +%H:%M:%S)] RUN slot=$SLOT idx=$i fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stockformer_faithful_eval \
    --phase biotech_nbi_enriched --fold "$FOLD" --seed "$SEED" \
    --output-dir-root "$RAW_ROOT" \
    --total-timesteps 10000 --pretrain-epochs 30 > "$LF" 2>&1 \
    && {
      SRC="$RAW_ROOT/biotech_nbi_enriched/fold${FOLD}_seed${SEED}.json"
      if [ -f "$SRC" ]; then
        mv -f "$SRC" "$OUT"
        echo "[OK] $OUT"
      else
        echo "[FAIL] $OUT (expected $SRC, not found)"
      fi
    } \
    || echo "[FAIL] $OUT ; see $LF"
done

echo "DONE slot=$SLOT $(date -Is)"
