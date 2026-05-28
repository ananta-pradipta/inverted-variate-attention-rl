#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=general
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

# Non-learning baselines (buy-and-hold, equal-weight, J-T momentum,
# reversal, vol-targeted market) on the InVAR-RL test segments. CPU-only,
# deterministic, ~5 min per fold. Runs all 5 folds in one job.
OUT_DIR="invar_rl/results/non_learning_baselines"
LOG_DIR="invar_rl/logs/non_learning_baselines"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is)"

for F in 1 2 3 4 5; do
  LF="${LOG_DIR}/fold${F}.log"
  echo "[$(date +%H:%M:%S)] non-learning eval fold=$F"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.non_learning_eval \
    --fold "$F" --segment test \
    --output-dir "$OUT_DIR" \
    --panel_kind lattice_native \
    --two_regime_val \
    > "$LF" 2>&1
done

echo "DONE $(date -Is)"
