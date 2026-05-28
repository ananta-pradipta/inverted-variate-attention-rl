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

# InVAR-RL Stage 3 RL ablation sweep. One job per (ABLATION, FOLD):
# sbatch --export=ALL,ABLATION=random_l1,FOLD=1 ...
# Supports ABLATION in {random_l1, equal_l2, stripped_l3, none}.
ABLATION="${ABLATION:?ABLATION must be set}"
FOLD="${FOLD:?FOLD must be set via --export=FOLD=<1..5>}"
METHODS="${METHODS:-recurrent_ppo,feedforward_ppo,sac}"
CKPT_DIR="invar_rl/results/stage1/_ckpt"
OUT_ROOT="invar_rl/results/stage3_rl_ablation"
OUT_DIR="${OUT_ROOT}/${ABLATION}"
LOG_DIR="invar_rl/logs/stage3_rl_ablation/${ABLATION}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) ablation=${ABLATION} fold=${FOLD} methods=${METHODS}"
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
  echo "[$(date +%H:%M:%S)] rl-ablation ablation=${ABLATION} fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage3_rl_ablation \
    --ablation "$ABLATION" \
    --fold "$FOLD" --seed "$SEED" \
    --layer1-ckpt "$CKPT" \
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

echo "DONE ablation=${ABLATION} fold=${FOLD} $(date -Is)"
