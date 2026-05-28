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

# Faithful FinRL reproduction. Phase 1: DJIA-30 credibility gate (train
# 2009-2015, test 2016-2020). Phase 2: stress test on universal S&P 500.
# --export=ALL,PHASE={djia,universal},METHOD={ppo,a2c,ddpg},SEED=k,FOLD=k
PHASE="${PHASE:-djia}"
METHOD="${METHOD:-ppo}"
SEED="${SEED:-42}"
FOLD="${FOLD:-1}"
TOTAL_TS="${TOTAL_TS:-50000}"
OUT_ROOT="invar_rl/results/finrl_faithful"
LOG_DIR="invar_rl/logs/finrl_faithful"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) phase=${PHASE} method=${METHOD} seed=${SEED}"
nvidia-smi | head -8 || true

if [ "$PHASE" = "universal" ]; then
  LF="${LOG_DIR}/${PHASE}_fold${FOLD}_${METHOD}_seed${SEED}.log"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.finrl_faithful_eval \
    --phase "$PHASE" \
    --method "$METHOD" \
    --seed "$SEED" \
    --fold "$FOLD" \
    --output-dir-root "$OUT_ROOT" \
    --total-timesteps "$TOTAL_TS" \
    > "$LF" 2>&1
else
  LF="${LOG_DIR}/${PHASE}_${METHOD}_seed${SEED}.log"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.finrl_faithful_eval \
    --phase "$PHASE" \
    --method "$METHOD" \
    --seed "$SEED" \
    --output-dir-root "$OUT_ROOT" \
    --total-timesteps "$TOTAL_TS" \
    > "$LF" 2>&1
fi

echo "DONE phase=${PHASE} method=${METHOD} seed=${SEED} $(date -Is)"
