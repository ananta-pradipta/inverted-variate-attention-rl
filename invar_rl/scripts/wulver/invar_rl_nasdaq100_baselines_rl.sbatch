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

# Phase 5.5 NASDAQ-100 whole-stack RL baselines (Panel B):
#   FinRL PPO/A2C/DDPG  : 3 algos x 5 folds x 5 seeds = 75 cells.
#   StockFormer faithful: 5 folds x 5 seeds = 25 cells.
# Total: 100 cells across 10 throttled slots.
#
# Submit (10 parallel slots):
#   for SLOT in 0 1 2 3 4 5 6 7 8 9; do
#     sbatch --export=ALL,SLOT=$SLOT,N_SLOTS=10 \
#       invar_rl/scripts/wulver/invar_rl_nasdaq100_baselines_rl.sbatch
#   done

set -uo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

SLOT="${SLOT:-0}"
N_SLOTS="${N_SLOTS:-10}"

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

# Outputs land under the canonical NDX rollup root so the Phase 5.5 rollup
# script can read FinRL + StockFormer alongside the Layer-1 ranker baselines.
OUT_FINRL_ROOT="outputs/nasdaq100/baselines"
OUT_SF_ROOT="outputs/nasdaq100/baselines/stockformer"
LOG_DIR="invar_rl/logs/nasdaq100_baselines_rl"
mkdir -p "$OUT_FINRL_ROOT/finrl_ppo" \
         "$OUT_FINRL_ROOT/finrl_a2c" \
         "$OUT_FINRL_ROOT/finrl_ddpg" \
         "$OUT_SF_ROOT" \
         "$LOG_DIR"

# Build full 100-cell list, deterministic order.
declare -a CELLS
IDX=0
# FinRL: 3 algos x 5 folds x 5 seeds = 75
for ALGO in ppo a2c ddpg; do
  for F in 1 2 3 4 5; do
    for SEED in 42 43 44 45 46; do
      CELLS[$IDX]="finrl $ALGO $F $SEED"; IDX=$((IDX+1))
    done
  done
done
# StockFormer: 5 folds x 5 seeds = 25
for F in 1 2 3 4 5; do
  for SEED in 42 43 44 45 46; do
    CELLS[$IDX]="stockformer - $F $SEED"; IDX=$((IDX+1))
  done
done

echo "host=$(hostname) slot=$SLOT/$N_SLOTS total_cells=${#CELLS[@]} start=$(date -Is)"
nvidia-smi | head -8 || true

for i in "${!CELLS[@]}"; do
  if [ $((i % N_SLOTS)) -ne "$SLOT" ]; then continue; fi
  read TYPE ALGO FOLD SEED <<< "${CELLS[$i]}"
  case "$TYPE" in
    finrl)
      OUT_DIR="$OUT_FINRL_ROOT/finrl_${ALGO}"
      # finrl writes fold${F}_${ALGO}_seed${SEED}.json natively. Rename to
      # fold${F}_seed${SEED}.json to match the Phase 5.5 rollup schema.
      OUT_RAW="$OUT_DIR/fold${FOLD}_${ALGO}_seed${SEED}.json"
      OUT="$OUT_DIR/fold${FOLD}_seed${SEED}.json"
      LF="$LOG_DIR/finrl_${ALGO}_fold${FOLD}_seed${SEED}.log" ;;
    stockformer)
      OUT_DIR="$OUT_SF_ROOT"
      OUT_RAW="$OUT_DIR/fold${FOLD}_seed${SEED}.json"
      OUT="$OUT_RAW"
      LF="$LOG_DIR/sf_fold${FOLD}_seed${SEED}.log" ;;
  esac
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] SKIP slot=$SLOT idx=$i $OUT exists"
    continue
  fi
  echo "[$(date +%H:%M:%S)] RUN slot=$SLOT idx=$i type=$TYPE algo=$ALGO fold=$FOLD seed=$SEED"
  case "$TYPE" in
    finrl)
      # The eval script writes the JSON named with the ALGO in the filename;
      # to fit the rollup schema, we drop the JSON next to the algo-keyed
      # output_dir AND rename atomically once produced.
      PYTHONPATH=$PWD python3 -u -m invar_rl.training.finrl_faithful_eval \
        --phase nasdaq100 --method "$ALGO" --fold "$FOLD" --seed "$SEED" \
        --output-dir-root "$OUT_FINRL_ROOT/_finrl_raw" \
        --total-timesteps 50000 > "$LF" 2>&1 \
        && {
          # The finrl eval routes outputs to $ROOT/$PHASE/foldF_ALGO_seedS.json.
          SRC="$OUT_FINRL_ROOT/_finrl_raw/nasdaq100/fold${FOLD}_${ALGO}_seed${SEED}.json"
          if [ -f "$SRC" ]; then
            mv -f "$SRC" "$OUT"
            echo "[OK] $OUT"
          else
            echo "[FAIL] $OUT (expected $SRC, not found)"
          fi
        } \
        || echo "[FAIL] $OUT ; see $LF" ;;
    stockformer)
      PYTHONPATH=$PWD python3 -u -m invar_rl.training.stockformer_faithful_eval \
        --phase nasdaq100 --fold "$FOLD" --seed "$SEED" \
        --output-dir-root "$OUT_FINRL_ROOT/_sf_raw" \
        --total-timesteps 10000 --pretrain-epochs 30 > "$LF" 2>&1 \
        && {
          SRC="$OUT_FINRL_ROOT/_sf_raw/nasdaq100/fold${FOLD}_seed${SEED}.json"
          if [ -f "$SRC" ]; then
            mv -f "$SRC" "$OUT"
            echo "[OK] $OUT"
          else
            echo "[FAIL] $OUT (expected $SRC, not found)"
          fi
        } \
        || echo "[FAIL] $OUT ; see $LF" ;;
  esac
done

echo "DONE slot=$SLOT $(date -Is)"
