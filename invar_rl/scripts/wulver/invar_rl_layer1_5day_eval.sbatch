#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=02:00:00
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

# Layer-1 5-day non-overlap top-25 L/S portfolio eval on the RAG-STAR
# Table 4 convention. One job runs canonical InVAR across all (fold,
# seed) cells. Use FOLD=k to restrict.
FOLD="${FOLD:-all}"
K_LONG="${K_LONG:-25}"
COST_BPS="${COST_BPS:-5.0}"
CKPT_DIR="invar_rl/results/stage1/_ckpt"
OUT_ROOT="invar_rl/results/layer1_5day"
LOG_DIR="invar_rl/logs/layer1_5day"
mkdir -p "$OUT_ROOT/canonical" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) FOLD=${FOLD} K=${K_LONG} COST=${COST_BPS}bps"
nvidia-smi | head -8 || true

FOLDS=(1 2 3 4 5)
if [ "$FOLD" != "all" ]; then FOLDS=("$FOLD"); fi

for F in "${FOLDS[@]}"; do
  for S in 42 43 44 45 46; do
    CKPT="${CKPT_DIR}/fold${F}_seed${S}_full.pt"
    OUT="${OUT_ROOT}/canonical/fold${F}_seed${S}.json"
    if [ ! -f "$CKPT" ]; then
      echo "[$(date +%H:%M:%S)] skip seed=$S fold=$F: ckpt missing"
      continue
    fi
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="${LOG_DIR}/canonical_fold${F}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] canonical fold=$F seed=$S"
    PYTHONPATH=$PWD python3 -u -m invar_rl.training.layer1_5day_eval \
      --source canonical \
      --fold "$F" --seed "$S" \
      --layer1-ckpt "$CKPT" \
      --output-dir-root "$OUT_ROOT" \
      --k "$K_LONG" \
      --cost-bps-roundtrip "$COST_BPS" \
      --panel_kind lattice_native \
      --two_regime_val \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
