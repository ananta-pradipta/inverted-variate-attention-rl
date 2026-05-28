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

# Phase 1 of fairness audit issue #5 fix: retrain FinRL whole-stack RL
# (PPO + A2C + DDPG) on the biotech NBI ENRICHED 22-feature panel.
# Grid: 3 algos x 5 folds x 5 seeds = 75 cells across 10 throttled slots.
#
# Submit (10 parallel slots):
#   for SLOT in 0 1 2 3 4 5 6 7 8 9; do
#     sbatch --export=ALL,SLOT=$SLOT,N_SLOTS=10 \
#       invar_rl/scripts/wulver/invar_rl_biotech_nbi_enriched_finrl.sbatch
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

OUT_ROOT="outputs/biotech_nbi_enriched/whole_stack_rl/finrl"
RAW_ROOT="${OUT_ROOT}/_raw"
LOG_DIR="invar_rl/logs/biotech_nbi_enriched_finrl"
mkdir -p "$OUT_ROOT/finrl_ppo" \
         "$OUT_ROOT/finrl_a2c" \
         "$OUT_ROOT/finrl_ddpg" \
         "$RAW_ROOT" \
         "$LOG_DIR"

# 3 algos x 5 folds x 5 seeds = 75
declare -a CELLS
IDX=0
for ALGO in ppo a2c ddpg; do
  for F in 1 2 3 4 5; do
    for SEED in 42 43 44 45 46; do
      CELLS[$IDX]="$ALGO $F $SEED"
      IDX=$((IDX + 1))
    done
  done
done

echo "host=$(hostname) slot=$SLOT/$N_SLOTS total_cells=${#CELLS[@]} start=$(date -Is)"
nvidia-smi | head -8 || true

for i in "${!CELLS[@]}"; do
  if [ $((i % N_SLOTS)) -ne "$SLOT" ]; then continue; fi
  read ALGO FOLD SEED <<< "${CELLS[$i]}"
  OUT_DIR="$OUT_ROOT/finrl_${ALGO}"
  OUT="$OUT_DIR/fold${FOLD}_seed${SEED}.json"
  LF="$LOG_DIR/finrl_${ALGO}_fold${FOLD}_seed${SEED}.log"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] SKIP slot=$SLOT idx=$i $OUT exists"
    continue
  fi
  echo "[$(date +%H:%M:%S)] RUN slot=$SLOT idx=$i algo=$ALGO fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.finrl_faithful_eval \
    --phase biotech_nbi_enriched --method "$ALGO" \
    --fold "$FOLD" --seed "$SEED" \
    --output-dir-root "$RAW_ROOT" \
    --total-timesteps 50000 > "$LF" 2>&1 \
    && {
      SRC="$RAW_ROOT/biotech_nbi_enriched/fold${FOLD}_${ALGO}_seed${SEED}.json"
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
