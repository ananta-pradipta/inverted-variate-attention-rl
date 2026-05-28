#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# DeepTrader (AAAI 2021) eval sweep.
# Default sweep: 5 DJIA credibility seeds + 25 universal cells = 30 cells.
# Run a single cell by overriding PHASE+FOLD+SEED via the env, otherwise
# the loop below executes the full sweep with skip-if-output-exists.

set -uo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

OUT_ROOT="${OUT_ROOT:-invar_rl/results/deeptrader}"
LOG_DIR="${LOG_DIR:-invar_rl/logs/deeptrader}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-500}"
UNIVERSE_K="${UNIVERSE_K:-30}"
mkdir -p "$OUT_ROOT/djia" "$OUT_ROOT/universal" "$LOG_DIR"

# Single-cell override: if SMOKE_PHASE is set, run exactly that one cell.
if [ -n "${SMOKE_PHASE:-}" ]; then
  S_PHASE="$SMOKE_PHASE"
  S_FOLD="${SMOKE_FOLD:-1}"
  S_SEED="${SMOKE_SEED:-42}"
  S_EPOCHS="${SMOKE_EPOCHS:-$TOTAL_EPOCHS}"
  if [ "$S_PHASE" = "universal" ]; then
    OUT="$OUT_ROOT/universal/fold${S_FOLD}_seed${S_SEED}.json"
    LF="$LOG_DIR/universal_fold${S_FOLD}_seed${S_SEED}.log"
  else
    OUT="$OUT_ROOT/djia/djia_seed${S_SEED}.json"
    LF="$LOG_DIR/djia_seed${S_SEED}.log"
  fi
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] SKIP smoke $OUT exists"
    exit 0
  fi
  echo "[$(date +%H:%M:%S)] SMOKE phase=$S_PHASE fold=$S_FOLD seed=$S_SEED epochs=$S_EPOCHS"
  nvidia-smi | head -8 || true
  if [ "$S_PHASE" = "universal" ]; then
    PYTHONPATH=$PWD python3 -u -m invar_rl.training.deeptrader_eval \
      --phase universal --fold "$S_FOLD" --seed "$S_SEED" \
      --output-dir-root "$OUT_ROOT" \
      --total-epochs "$S_EPOCHS" --universe-k "$UNIVERSE_K" \
      > "$LF" 2>&1 \
      && echo "[OK] $OUT" || echo "[FAIL] $OUT"
  else
    PYTHONPATH=$PWD python3 -u -m invar_rl.training.deeptrader_eval \
      --phase djia --seed "$S_SEED" \
      --output-dir-root "$OUT_ROOT" \
      --total-epochs "$S_EPOCHS" \
      > "$LF" 2>&1 \
      && echo "[OK] $OUT" || echo "[FAIL] $OUT"
  fi
  echo "DONE smoke $(date -Is)"
  exit 0
fi

# Full sweep: 5 DJIA credibility seeds + 25 universal cells.
echo "host=$(hostname) start=$(date -Is) total_epochs=$TOTAL_EPOCHS"
nvidia-smi | head -8 || true

# SEEDS env var lets us 5-slot-partition the sweep across parallel jobs
# (one job per seed; each job loops the seed's DJIA + 5 universal folds).
SEEDS="${SEEDS:-42 43 44 45 46}"

for SEED in $SEEDS; do
  OUT_DJIA="$OUT_ROOT/djia/djia_seed${SEED}.json"
  LF_DJIA="$LOG_DIR/djia_seed${SEED}.log"
  if [ -f "$OUT_DJIA" ]; then
    echo "[$(date +%H:%M:%S)] SKIP djia seed=$SEED ($OUT_DJIA exists)"
  else
    echo "[$(date +%H:%M:%S)] RUN djia seed=$SEED"
    PYTHONPATH=$PWD python3 -u -m invar_rl.training.deeptrader_eval \
      --phase djia --seed "$SEED" \
      --output-dir-root "$OUT_ROOT" \
      --total-epochs "$TOTAL_EPOCHS" \
      > "$LF_DJIA" 2>&1 \
      && echo "[OK] $OUT_DJIA" || echo "[FAIL] $OUT_DJIA"
  fi
  for F in 1 2 3 4 5; do
    OUT_UNI="$OUT_ROOT/universal/fold${F}_seed${SEED}.json"
    LF_UNI="$LOG_DIR/universal_fold${F}_seed${SEED}.log"
    if [ -f "$OUT_UNI" ]; then
      echo "[$(date +%H:%M:%S)] SKIP universal fold=$F seed=$SEED ($OUT_UNI exists)"
      continue
    fi
    echo "[$(date +%H:%M:%S)] RUN universal fold=$F seed=$SEED"
    PYTHONPATH=$PWD python3 -u -m invar_rl.training.deeptrader_eval \
      --phase universal --fold "$F" --seed "$SEED" \
      --output-dir-root "$OUT_ROOT" \
      --total-epochs "$TOTAL_EPOCHS" --universe-k "$UNIVERSE_K" \
      > "$LF_UNI" 2>&1 \
      && echo "[OK] $OUT_UNI" || echo "[FAIL] $OUT_UNI"
  done
done

echo "DONE $(date -Is)"
