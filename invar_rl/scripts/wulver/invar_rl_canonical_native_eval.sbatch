#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=01:00:00
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

# Canonical InVAR top-K L/S daily portfolio eval (Panel A row).
# One job; loops all 25 (fold, seed) cells.
CKPT_DIR="invar_rl/results/stage1/_ckpt"
OUT_DIR="invar_rl/results/native_ranker_baselines/canonical_invar"
LOG_DIR="invar_rl/logs/canonical_native_eval"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is)"

for F in 1 2 3 4 5; do
  for S in 42 43 44 45 46; do
    CKPT="${CKPT_DIR}/fold${F}_seed${S}_full.pt"
    OUT="${OUT_DIR}/fold${F}_seed${S}.json"
    if [ ! -f "$CKPT" ]; then
      echo "[$(date +%H:%M:%S)] skip seed=$S fold=$F: ckpt missing"
      continue
    fi
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="${LOG_DIR}/fold${F}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] canonical fold=$F seed=$S"
    PYTHONPATH=$PWD python3 -u -m invar_rl.training.canonical_native_eval \
      --fold "$F" --seed "$S" \
      --layer1-ckpt "$CKPT" \
      --output-dir "$OUT_DIR" \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
