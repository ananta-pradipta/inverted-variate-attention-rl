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

# Option C (diff_sharpe) Phase 2 L2L3 evaluation on SP500. Mirrors
# the canonical robust_phase1_l2l3 sbatch: wrapper eval (equal-weight
# K=50) + SAC L/S (K=50) on the diff_sharpe-trained Layer 1 ckpts.
# 5 parallel slots, each handling a round-robin subset of the 25 cells.
# Daily tape capture is ON so the bootstrap p-value pipeline has the
# per-day return series.
#
# Single sbatch per [[feedback_wulver_single_job]]; generic job name
# per [[feedback_wulver_job_names]].
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_sp500_diff_sharpe_l2l3_phase2.sbatch
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

CKPT_ROOT="outputs/sp500/layer1_diff_sharpe_phase2/_ckpt"
OUT_ROOT="outputs/sp500/stage3_rl_ablation/diff_sharpe_phase2"
TAPE_ROOT="outputs/sp500/stage3_rl_ablation/diff_sharpe_phase2/daily_tape"
LOG_ROOT="invar_rl/logs/sp500_diff_sharpe_l2l3_phase2"
mkdir -p "$OUT_ROOT" "$TAPE_ROOT" "$LOG_ROOT"

# 25 (fold, seed) tuples; round-robin over 5 slots so each slot owns 5 cells.
TUPLES=()
for FOLD in 1 2 3 4 5; do
  for SEED in 42 43 44 45 46; do
    TUPLES+=("${FOLD}|${SEED}")
  done
done
N_TUPLES=${#TUPLES[@]}
N_SLOTS=5

echo "host=$(hostname) start=$(date -Is) slot=$SLOT total_tuples=$N_TUPLES"
nvidia-smi | head -8 || true

for i in $(seq 0 $((N_TUPLES - 1))); do
  if [ $((i % N_SLOTS)) -ne "$SLOT" ]; then
    continue
  fi
  IFS='|' read -r FOLD SEED <<< "${TUPLES[$i]}"
  CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
  OUT_JSON="${OUT_ROOT}/fold${FOLD}_seed${SEED}.json"
  LF="${LOG_ROOT}/fold${FOLD}_seed${SEED}.log"

  if [ ! -f "$CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED: ckpt missing $CKPT"
    continue
  fi
  if [ -f "$OUT_JSON" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED (json present)"
    continue
  fi
  echo "[$(date +%H:%M:%S)] run l2l3 fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage3_rl_ablation \
    --ablation equal_l2 \
    --fold "$FOLD" --seed "$SEED" \
    --layer1-ckpt "$CKPT" \
    --output-dir "$OUT_ROOT" \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --equal-topk-k 50 \
    --methods sac \
    --save-daily-tape \
    --daily-tape-dir "$TAPE_ROOT" \
    > "$LF" 2>&1 || {
      echo "FAILED fold=$FOLD seed=$SEED; see $LF"
      tail -50 "$LF"
      continue
    }
done

echo "DONE slot=${SLOT} $(date -Is)"
ls -la "$OUT_ROOT/" 2>/dev/null || true
