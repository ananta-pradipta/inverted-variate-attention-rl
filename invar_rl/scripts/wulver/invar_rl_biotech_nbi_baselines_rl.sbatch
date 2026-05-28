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

# Phase 5.5 biotech NBI whole-stack RL baselines (Panel B):
#   FinRL PPO/A2C/DDPG  : 3 algos x 5 folds x 5 seeds = 75 cells.
#   StockFormer faithful: 5 folds x 5 seeds = 25 cells.
# Total: 100 cells across 10 throttled slots.
#
# Submit (10 parallel slots):
#   for SLOT in 0 1 2 3 4 5 6 7 8 9; do
#     sbatch --export=ALL,SLOT=$SLOT,N_SLOTS=10 \
#       invar_rl/scripts/wulver/invar_rl_biotech_nbi_baselines_rl.sbatch
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

OUT_FINRL_ROOT="outputs/biotech_nbi/baselines"
OUT_SF_ROOT="outputs/biotech_nbi/baselines/stockformer"
LOG_DIR="invar_rl/logs/biotech_nbi_baselines_rl"
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
      OUT="$OUT_DIR/fold${FOLD}_seed${SEED}.json"
      LF="$LOG_DIR/finrl_${ALGO}_fold${FOLD}_seed${SEED}.log" ;;
    stockformer)
      OUT_DIR="$OUT_SF_ROOT"
      OUT="$OUT_DIR/fold${FOLD}_seed${SEED}.json"
      LF="$LOG_DIR/sf_fold${FOLD}_seed${SEED}.log" ;;
  esac
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] SKIP slot=$SLOT idx=$i $OUT exists"
    continue
  fi
  echo "[$(date +%H:%M:%S)] RUN slot=$SLOT idx=$i type=$TYPE algo=$ALGO fold=$FOLD seed=$SEED"
  case "$TYPE" in
    finrl)
      PYTHONPATH=$PWD python3 -u -m invar_rl.training.finrl_faithful_eval \
        --phase biotech_nbi --method "$ALGO" --fold "$FOLD" --seed "$SEED" \
        --output-dir-root "$OUT_FINRL_ROOT/_finrl_raw" \
        --total-timesteps 50000 > "$LF" 2>&1 \
        && {
          SRC="$OUT_FINRL_ROOT/_finrl_raw/biotech_nbi/fold${FOLD}_${ALGO}_seed${SEED}.json"
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
        --phase biotech_nbi --fold "$FOLD" --seed "$SEED" \
        --output-dir-root "$OUT_FINRL_ROOT/_sf_raw" \
        --total-timesteps 10000 --pretrain-epochs 30 > "$LF" 2>&1 \
        && {
          SRC="$OUT_FINRL_ROOT/_sf_raw/biotech_nbi/fold${FOLD}_seed${SEED}.json"
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
