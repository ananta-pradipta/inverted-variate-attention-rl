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

PHASE="${PHASE:-nasdaq}"
SEED="${SEED:-42}"
FOLD="${FOLD:-1}"
TOTAL_TS="${TOTAL_TS:-30000}"
PRETRAIN="${PRETRAIN:-50}"
OUT_ROOT="invar_rl/results/stockformer_faithful"
LOG_DIR="invar_rl/logs/stockformer_faithful"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) phase=${PHASE} seed=${SEED} fold=${FOLD}"
nvidia-smi | head -8 || true

if [ "$PHASE" = "universal" ]; then
  LF="${LOG_DIR}/${PHASE}_fold${FOLD}_seed${SEED}.log"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stockformer_faithful_eval \
    --phase "$PHASE" --seed "$SEED" --fold "$FOLD" \
    --output-dir-root "$OUT_ROOT" \
    --total-timesteps "$TOTAL_TS" \
    --pretrain-epochs "$PRETRAIN" \
    > "$LF" 2>&1
else
  LF="${LOG_DIR}/${PHASE}_seed${SEED}.log"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stockformer_faithful_eval \
    --phase "$PHASE" --seed "$SEED" \
    --output-dir-root "$OUT_ROOT" \
    --total-timesteps "$TOTAL_TS" \
    --pretrain-epochs "$PRETRAIN" \
    > "$LF" 2>&1
fi

echo "DONE phase=${PHASE} seed=${SEED} $(date -Is)"
