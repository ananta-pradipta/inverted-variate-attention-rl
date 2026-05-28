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

# InVAR-RL Stage 3 RL with EXTERNAL Layer-1 baseline. Swap-L1 whole-stack
# comparison: same Layer 2 (cvxpy MV-QP) + same Layer 3 (recurrent_ppo,
# feedforward_ppo, sac); only Layer 1 differs.
# --export=ALL,BASELINE={master,factorvae,itransformer,stockmixer,dystage,mera,swa_invar},FOLD={1..5}
BASELINE="${BASELINE:?BASELINE must be set}"
FOLD="${FOLD:?FOLD must be set}"
METHODS="${METHODS:-recurrent_ppo,feedforward_ppo,sac}"
OUT_ROOT="invar_rl/results/stage3_rl_external"
OUT_DIR="${OUT_ROOT}/${BASELINE}"
LOG_DIR="invar_rl/logs/stage3_rl_external/${BASELINE}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) baseline=${BASELINE} fold=${FOLD} methods=${METHODS}"
nvidia-smi | head -8 || true

run_seed() {
  local SEED="$1"
  local OUT="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT (already done)"
    return
  fi
  local LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] external-L1=${BASELINE} fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage3_rl_external \
    --baseline "$BASELINE" \
    --fold "$FOLD" --seed "$SEED" \
    --layer2 invar_rl/configs/layer2.yaml \
    --layer3 invar_rl/configs/layer3.yaml \
    --stage3 invar_rl/configs/stage3.yaml \
    --output-dir-root "$OUT_ROOT" \
    --methods "$METHODS" \
    --panel_kind lattice_native \
    --two_regime_val \
    > "$LF" 2>&1
}

for S in 42 43 44 45 46; do
  run_seed "$S"
done

echo "DONE baseline=${BASELINE} fold=${FOLD} $(date -Is)"
