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

# Option C (DSL) Phase 2 L2L3 evaluation on NASDAQ-100. Reuses the
# canonical NDX-100 Phase 5 SAC pipeline (invar_rl.training.nasdaq100_layer3_sac)
# byte-for-byte, only pointing CKPT_ROOT at the DSL-trained Layer 1
# ckpts in outputs/nasdaq100/layer1_diff_sharpe_phase2/_ckpt/.
#
# Reference: canonical NDX-100 Phase 5 L/S pool +1.194 (computed from
# outputs/nasdaq100/layer3/ls/summary/). The DSL Phase 2 L2L3 outputs
# under outputs/nasdaq100/stage3_rl_ablation/diff_sharpe_phase2/ are
# diffable against that reference.
#
# 5 slots (mirrors SP500 DSL L2L3 SLOT layout): each slot owns 5 of
# the 25 (fold, seed) tuples via round-robin. Single sbatch per
# [[feedback_wulver_single_job]]; generic job name per
# [[feedback_wulver_job_names]].
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_nasdaq100_diff_sharpe_l2l3_phase2.sbatch
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
PROTO="${PROTO:-ls}"
TIMESTEPS="${TIMESTEPS:-20000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"

CKPT_ROOT="outputs/nasdaq100/layer1_diff_sharpe_phase2/_ckpt"
OUT_ROOT="outputs/nasdaq100/stage3_rl_ablation/diff_sharpe_phase2"
LOG_ROOT="invar_rl/logs/nasdaq100_diff_sharpe_l2l3_phase2"
mkdir -p "${OUT_ROOT}/${PROTO}/summary" "$LOG_ROOT"

# 25 (fold, seed) tuples; round-robin over 5 slots so each slot owns 5 cells.
TUPLES=()
for FOLD in 1 2 3 4 5; do
  for SEED in 42 43 44 45 46; do
    TUPLES+=("${FOLD}|${SEED}")
  done
done
N_TUPLES=${#TUPLES[@]}
N_SLOTS=5

echo "host=$(hostname) start=$(date -Is) slot=$SLOT proto=$PROTO total_tuples=$N_TUPLES"
nvidia-smi | head -8 || true

for i in $(seq 0 $((N_TUPLES - 1))); do
  if [ $((i % N_SLOTS)) -ne "$SLOT" ]; then
    continue
  fi
  IFS='|' read -r FOLD SEED <<< "${TUPLES[$i]}"
  CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
  OUT_PARQ="${OUT_ROOT}/${PROTO}/fold${FOLD}_seed${SEED}.parquet"
  SUMMARY="${OUT_ROOT}/${PROTO}/summary/fold${FOLD}_seed${SEED}.json"
  LF="${LOG_ROOT}/fold${FOLD}_seed${SEED}_${PROTO}.log"

  if [ ! -f "$CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED: ckpt missing $CKPT"
    continue
  fi
  if [ -f "$OUT_PARQ" ] && [ -f "$SUMMARY" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED (outputs exist)"
    continue
  fi
  echo "[$(date +%H:%M:%S)] sac-train fold=$FOLD seed=$SEED proto=$PROTO"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.nasdaq100_layer3_sac \
    --fold "$FOLD" --seed "$SEED" --protocol "$PROTO" \
    --total-timesteps "$TIMESTEPS" \
    --eval-freq "$EVAL_FREQ" \
    --output-dir-root "$OUT_ROOT" \
    --layer1-ckpt-root "$CKPT_ROOT" \
    --layer3 invar_rl/configs/layer3.yaml \
    --stage3 invar_rl/configs/stage3.yaml \
    > "$LF" 2>&1 || {
      echo "FAILED fold=$FOLD seed=$SEED ; see $LF"
      tail -50 "$LF"
      continue
    }
done

echo "DONE slot=${SLOT} proto=$PROTO $(date -Is)"
ls -la "${OUT_ROOT}/${PROTO}/summary/" 2>/dev/null || true
