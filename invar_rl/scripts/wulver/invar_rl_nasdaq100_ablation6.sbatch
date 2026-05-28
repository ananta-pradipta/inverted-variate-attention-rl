#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# NASDAQ-100 Ablation 6: macro-conditional concentration (2-D SAC action,
# K_t in [3, 20]). Cross-universe replication of the SP500 Ablation 6
# sbatch (see invar_rl/scripts/wulver/invar_rl_sp500_ablation6.sbatch).
# ONE sbatch slot serially processes a fixed shard of the 25 (fold, seed)
# grid; 5 parallel slots span the whole grid. Single-job-per-sbatch per
# [[feedback_wulver_single_job]]; generic job name per
# [[feedback_wulver_job_names]].
#
# Grid:
#   FOLDS = 1..5
#   SEEDS = 42, 43, 44, 45, 46
#   total: 25 cells
#
# Sharding: round-robin (i % N_SLOTS == SLOT), 5 cells per slot.
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_nasdaq100_ablation6.sbatch
#   done

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

SLOT="${SLOT:?SLOT must be set via --export=SLOT=0..4}"
TIMESTEPS="${TIMESTEPS:-20000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"

OUT_ROOT="outputs/nasdaq100/ablations/ablation6"
LOG_ROOT="invar_rl/logs/nasdaq100_ablation6"
CKPT_ROOT="outputs/nasdaq100/layer1/_ckpt"
mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$OUT_ROOT/summary"

# Build the (fold, seed) tuple list. 25 entries.
TUPLES=()
for FOLD in 1 2 3 4 5; do
  for SEED in 42 43 44 45 46; do
    TUPLES+=("${FOLD}|${SEED}")
  done
done
N_TUPLES=${#TUPLES[@]}    # 25
N_SLOTS=5

echo "host=$(hostname) start=$(date -Is) slot=$SLOT total_tuples=$N_TUPLES"
nvidia-smi | head -8 || true

# Round-robin sharding: this slot handles tuples i where i % N_SLOTS == SLOT.
for i in $(seq 0 $((N_TUPLES - 1))); do
  if [ $((i % N_SLOTS)) -ne "$SLOT" ]; then
    continue
  fi
  IFS='|' read -r FOLD SEED <<< "${TUPLES[$i]}"
  CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
  OUT="${OUT_ROOT}/fold${FOLD}_seed${SEED}.parquet"
  SUMMARY="${OUT_ROOT}/summary/fold${FOLD}_seed${SEED}.json"
  LF="${LOG_ROOT}/fold${FOLD}_seed${SEED}.log"

  if [ ! -f "$CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED: ckpt missing $CKPT"
    continue
  fi
  if [ -f "$OUT" ] && [ -f "$SUMMARY" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED (outputs exist)"
    continue
  fi
  echo "[$(date +%H:%M:%S)] run ablation6 fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.nasdaq100_layer3_sac_ablation6 \
    --fold "$FOLD" --seed "$SEED" \
    --total-timesteps "$TIMESTEPS" \
    --eval-freq "$EVAL_FREQ" \
    --output-dir-root "$OUT_ROOT" \
    --layer1-ckpt-root "$CKPT_ROOT" \
    --layer3 invar_rl/configs/layer3.yaml \
    --stage3 invar_rl/configs/stage3.yaml \
    > "$LF" 2>&1 || {
      echo "FAILED ablation6 fold=$FOLD seed=$SEED; see $LF"
      continue
    }
done

echo "DONE slot=$SLOT $(date -Is)"
