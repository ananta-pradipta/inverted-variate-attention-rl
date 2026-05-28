#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Phase 5.5 NASDAQ-100 Layer-1 ranker baselines (5 baselines x 25 cells = 125).
# 5-slot throttled wrapper per [[feedback_wulver_single_job]]. Generic job
# name per [[feedback_wulver_job_names]]. Skip-if-output-exists on the npz.
#
# Baselines: MASTER (AAAI'24), FactorVAE (AAAI'22), StockMixer (AAAI'24),
# DySTAGE (ICAIF'24), SWA-InVAR (this work).
# Per-cell training writes a predictions npz to:
#   results/baselines_nasdaq100_two_regime_val/{baseline}/fold{F}_seed{S}_predictions.npz
# After all 125 cells finish, run the NDX baseline eval harness to produce
#   outputs/nasdaq100/baselines/{baseline}/fold{F}_seed{S}.json
# with BOTH sharpe_ls (top-25 L/S wrapper) and sharpe_lo_native (native top-K).
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT,N_SLOTS=5 \
#       invar_rl/scripts/wulver/invar_rl_nasdaq100_baselines_layer1.sbatch
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

NPZ_ROOT="results/baselines_nasdaq100_two_regime_val"
EVAL_OUT="outputs/nasdaq100/baselines"
LOG_DIR="invar_rl/logs/nasdaq100_baselines_layer1"
mkdir -p "$NPZ_ROOT" "$EVAL_OUT" "$LOG_DIR"

# Build full 125-cell list (5 baselines x 5 folds x 5 seeds), deterministic
# order. Each baseline is invoked via its v2 training driver; the driver
# writes the predictions npz that the NDX baseline eval harness later
# consumes.
declare -a CELLS
IDX=0
for B in master factorvae stockmixer dystage swa_invar; do
  for F in 1 2 3 4 5; do
    for SEED in 42 43 44 45 46; do
      CELLS[$IDX]="$B $F $SEED"; IDX=$((IDX+1))
    done
  done
done

echo "host=$(hostname) slot=$SLOT/$N_SLOTS total_cells=${#CELLS[@]} start=$(date -Is)"
nvidia-smi | head -8 || true

for i in "${!CELLS[@]}"; do
  if [ $((i % N_SLOTS)) -ne "$SLOT" ]; then continue; fi
  read B F SEED <<< "${CELLS[$i]}"
  B_DIR="$NPZ_ROOT/$B"
  NPZ="$B_DIR/fold${F}_seed${SEED}_predictions.npz"
  LF="$LOG_DIR/${B}_fold${F}_seed${SEED}.log"
  if [ -f "$NPZ" ]; then
    echo "[$(date +%H:%M:%S)] SKIP slot=$SLOT idx=$i $NPZ exists"
    continue
  fi
  echo "[$(date +%H:%M:%S)] RUN slot=$SLOT idx=$i baseline=$B fold=$F seed=$SEED"
  mkdir -p "$B_DIR"
  case "$B" in
    master)
      PYTHONPATH=$PWD python3 -u -m src.baselines.train_master_v2 \
        --panel_kind nasdaq100 --two_regime_val \
        --fold "$F" --seed "$SEED" \
        --output_dir "$B_DIR" > "$LF" 2>&1 \
        && echo "[OK] $NPZ" || echo "[FAIL] $NPZ ; see $LF" ;;
    factorvae)
      PYTHONPATH=$PWD python3 -u -m src.baselines.train_factorvae_v2 \
        --panel_kind nasdaq100 --two_regime_val \
        --fold "$F" --seed "$SEED" \
        --output_dir "$B_DIR" > "$LF" 2>&1 \
        && echo "[OK] $NPZ" || echo "[FAIL] $NPZ ; see $LF" ;;
    stockmixer)
      PYTHONPATH=$PWD python3 -u -m src.baselines.train_stockmixer_v2 \
        --panel_kind nasdaq100 --two_regime_val \
        --fold "$F" --seed "$SEED" \
        --output_dir "$B_DIR" > "$LF" 2>&1 \
        && echo "[OK] $NPZ" || echo "[FAIL] $NPZ ; see $LF" ;;
    dystage)
      PYTHONPATH=$PWD python3 -u -m src.baselines.train_dystage_v2 \
        --panel_kind nasdaq100 --two_regime_val \
        --fold "$F" --seed "$SEED" \
        --output_dir "$B_DIR" > "$LF" 2>&1 \
        && echo "[OK] $NPZ" || echo "[FAIL] $NPZ ; see $LF" ;;
    swa_invar)
      PYTHONPATH=$PWD python3 -u -m src.baselines.train_swa_invar_v2 \
        --panel_kind nasdaq100 --two_regime_val \
        --fold "$F" --seed "$SEED" \
        --output_dir "$B_DIR" > "$LF" 2>&1 \
        && echo "[OK] $NPZ" || echo "[FAIL] $NPZ ; see $LF" ;;
  esac

  # Generate the per-cell wrapper+native eval JSON immediately after the
  # training cell completes. Uses --seed (not --sweep-fold) so this slot
  # only writes its own cell.
  if [ -f "$NPZ" ]; then
    EVAL_LF="$LOG_DIR/${B}_eval_fold${F}_seed${SEED}.log"
    PYTHONPATH=$PWD python3 -u -m invar_rl.training.nasdaq100_baseline_eval \
      --baseline "$B" --fold "$F" --seed "$SEED" \
      --npz-root "$NPZ_ROOT" \
      --output-dir-root "$EVAL_OUT" \
      --two_regime_val > "$EVAL_LF" 2>&1 \
      && echo "[EVAL OK] $B fold=$F seed=$SEED" \
      || echo "[EVAL FAIL] $B fold=$F seed=$SEED ; see $EVAL_LF"
  fi
done

# Slot 0 also runs the pooled rollup once. Other slots no-op.
if [ "$SLOT" = "0" ]; then
  ROLLUP_LF="$LOG_DIR/rollup.log"
  echo "[$(date +%H:%M:%S)] running pooled rollup (slot 0 only)"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.nasdaq100_baseline_eval \
    --rollup \
    --output-dir-root "$EVAL_OUT" > "$ROLLUP_LF" 2>&1 \
    && echo "[ROLLUP OK]" \
    || echo "[ROLLUP FAIL] see $ROLLUP_LF"
fi

echo "DONE slot=$SLOT $(date -Is)"
