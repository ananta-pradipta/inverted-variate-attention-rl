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

OUT_FINRL="invar_rl/results/finrl_faithful"
OUT_SF="invar_rl/results/stockformer_faithful"
LOG_FINRL="invar_rl/logs/finrl_faithful"
LOG_SF="invar_rl/logs/stockformer_faithful"
mkdir -p "$OUT_FINRL/djia" "$OUT_FINRL/universal" \
         "$OUT_SF/nasdaq" "$OUT_SF/universal" \
         "$LOG_FINRL" "$LOG_SF"

# Build full 120-cell list (deterministic order)
declare -a CELLS
IDX=0
# FinRL DJIA: 3 algos x 5 seeds = 15
for ALGO in ppo a2c ddpg; do
  for SEED in 42 43 44 45 46; do
    CELLS[$IDX]="finrl_djia $ALGO 0 $SEED"; IDX=$((IDX+1))
  done
done
# FinRL universal: 3 algos x 5 folds x 5 seeds = 75
for ALGO in ppo a2c ddpg; do
  for F in 1 2 3 4 5; do
    for SEED in 42 43 44 45 46; do
      CELLS[$IDX]="finrl_uni $ALGO $F $SEED"; IDX=$((IDX+1))
    done
  done
done
# StockFormer NASDAQ-86: 5 cells
for SEED in 42 43 44 45 46; do
  CELLS[$IDX]="sf_nasdaq - 0 $SEED"; IDX=$((IDX+1))
done
# StockFormer universal: 5 folds x 5 seeds = 25
for F in 1 2 3 4 5; do
  for SEED in 42 43 44 45 46; do
    CELLS[$IDX]="sf_uni - $F $SEED"; IDX=$((IDX+1))
  done
done

echo "host=$(hostname) slot=$SLOT/$N_SLOTS total_cells=${#CELLS[@]} start=$(date -Is)"
nvidia-smi | head -8 || true

for i in "${!CELLS[@]}"; do
  if [ $((i % N_SLOTS)) -ne "$SLOT" ]; then continue; fi
  read TYPE ALGO FOLD SEED <<< "${CELLS[$i]}"
  case "$TYPE" in
    finrl_djia)
      OUT="$OUT_FINRL/djia/djia_${ALGO}_seed${SEED}.json"
      LF="$LOG_FINRL/djia_${ALGO}_seed${SEED}.log" ;;
    finrl_uni)
      OUT="$OUT_FINRL/universal/fold${FOLD}_${ALGO}_seed${SEED}.json"
      LF="$LOG_FINRL/universal_fold${FOLD}_${ALGO}_seed${SEED}.log" ;;
    sf_nasdaq)
      OUT="$OUT_SF/nasdaq/nasdaq_seed${SEED}.json"
      LF="$LOG_SF/nasdaq_seed${SEED}.log" ;;
    sf_uni)
      OUT="$OUT_SF/universal/fold${FOLD}_seed${SEED}.json"
      LF="$LOG_SF/universal_fold${FOLD}_seed${SEED}.log" ;;
  esac
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] SKIP slot=$SLOT idx=$i $OUT exists"
    continue
  fi
  echo "[$(date +%H:%M:%S)] RUN slot=$SLOT idx=$i type=$TYPE algo=$ALGO fold=$FOLD seed=$SEED"
  case "$TYPE" in
    finrl_djia)
      PYTHONPATH=$PWD python3 -u -m invar_rl.training.finrl_faithful_eval \
        --phase djia --method "$ALGO" --seed "$SEED" \
        --output-dir-root "$OUT_FINRL" \
        --total-timesteps 50000 > "$LF" 2>&1 \
        && echo "[OK] $OUT" || echo "[FAIL] $OUT" ;;
    finrl_uni)
      PYTHONPATH=$PWD python3 -u -m invar_rl.training.finrl_faithful_eval \
        --phase universal --method "$ALGO" --fold "$FOLD" --seed "$SEED" \
        --output-dir-root "$OUT_FINRL" \
        --total-timesteps 50000 > "$LF" 2>&1 \
        && echo "[OK] $OUT" || echo "[FAIL] $OUT" ;;
    sf_nasdaq)
      PYTHONPATH=$PWD python3 -u -m invar_rl.training.stockformer_faithful_eval \
        --phase nasdaq --seed "$SEED" \
        --output-dir-root "$OUT_SF" \
        --total-timesteps 10000 --pretrain-epochs 50 > "$LF" 2>&1 \
        && echo "[OK] $OUT" || echo "[FAIL] $OUT" ;;
    sf_uni)
      PYTHONPATH=$PWD python3 -u -m invar_rl.training.stockformer_faithful_eval \
        --phase universal --fold "$FOLD" --seed "$SEED" \
        --output-dir-root "$OUT_SF" \
        --total-timesteps 10000 --pretrain-epochs 30 > "$LF" 2>&1 \
        && echo "[OK] $OUT" || echo "[FAIL] $OUT" ;;
  esac
done

echo "DONE slot=$SLOT $(date -Is)"
