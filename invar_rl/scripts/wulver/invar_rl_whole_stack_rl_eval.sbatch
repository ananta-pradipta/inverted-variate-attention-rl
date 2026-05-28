#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

# Whole-stack native RL baseline. One job per (BASELINE, FOLD), internal
# seed loop. BASELINE in {finrl, stockformer}.
BASELINE="${BASELINE:?BASELINE must be set}"
FOLD="${FOLD:?FOLD must be set}"
TOTAL_TS="${TOTAL_TS:-20000}"
UNIVERSE_K="${UNIVERSE_K:-50}"
FEATURE_SET="${FEATURE_SET:-lite}"
OUT_SUFFIX="${OUT_SUFFIX:-}"
OUT_ROOT="invar_rl/results/whole_stack_rl${OUT_SUFFIX}"
OUT_DIR="${OUT_ROOT}/${BASELINE}"
LOG_DIR="invar_rl/logs/whole_stack_rl${OUT_SUFFIX}/${BASELINE}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) baseline=${BASELINE} fold=${FOLD}"
nvidia-smi | head -8 || true

run_seed() {
  local SEED="$1"
  local OUT="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    return
  fi
  local LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] ${BASELINE} fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.whole_stack_rl_eval \
    --baseline "$BASELINE" \
    --fold "$FOLD" --seed "$SEED" \
    --output-dir-root "$OUT_ROOT" \
    --panel_kind lattice_native \
    --two_regime_val \
    --total-timesteps "$TOTAL_TS" \
    --universe-k "$UNIVERSE_K" \
    --feature-set "$FEATURE_SET" \
    > "$LF" 2>&1
}

for S in 42 43 44 45 46; do
  run_seed "$S"
done

echo "DONE ${BASELINE} fold=${FOLD} $(date -Is)"
