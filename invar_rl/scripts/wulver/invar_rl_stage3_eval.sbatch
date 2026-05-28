#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=03:00:00
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

# InVAR-RL Stage 3 eval: precompute layer-1 + layer-2 tapes through
# canonical InVAR + direct cvxpy QP, then run the 3 non-RL layer-3
# baselines (constant_full, vol_target, myopic_head) on the test
# segment. One job per FOLD via --export=FOLD=k, internal loop over
# the 5 seeds.
FOLD="${FOLD:?FOLD must be set via --export=FOLD=<1..5>}"
CKPT_DIR="invar_rl/results/stage1/_ckpt"
OUT_DIR="invar_rl/results/stage3_eval"
LOG_DIR="invar_rl/logs/stage3_eval"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD}"
nvidia-smi | head -8 || true

run_seed() {
  local SEED="$1"
  local CKPT="${CKPT_DIR}/fold${FOLD}_seed${SEED}_full.pt"
  local OUT="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ ! -f "$CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip seed=$SEED: ckpt missing $CKPT"
    return
  fi
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT (already done)"
    return
  fi
  local LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] eval fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage3_eval \
    --fold "$FOLD" --seed "$SEED" \
    --layer1-ckpt "$CKPT" \
    --layer2 invar_rl/configs/layer2.yaml \
    --layer3 invar_rl/configs/layer3.yaml \
    --stage3 invar_rl/configs/stage3.yaml \
    --output-dir "$OUT_DIR" \
    --panel_kind lattice_native \
    --two_regime_val \
    > "$LF" 2>&1
}

for S in 42 43 44 45 46; do
  run_seed "$S"
done

echo "DONE fold=${FOLD} $(date -Is)"
