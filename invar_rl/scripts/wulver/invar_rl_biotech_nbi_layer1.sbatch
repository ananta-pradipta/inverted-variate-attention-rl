#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# biotech NBI Phase 3 Layer 1: canonical InVAR (bankless + clpretrain)
# training. One sbatch slot trains 5 folds for ONE seed; submit 5
# instances with SLOT=0..4 to span seeds 42..46 in parallel
# (single-job-per-sbatch per [[feedback_wulver_single_job]]). Skip-if-
# output-exists guard via the per-cell metrics JSON.
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_biotech_nbi_layer1.sbatch
#   done

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

SLOT="${SLOT:-0}"
SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$SLOT]}"

OUT_DIR="outputs/biotech_nbi/layer1"
LOG_DIR="invar_rl/logs/biotech_nbi_layer1"
mkdir -p "$OUT_DIR/metrics" "$LOG_DIR"

echo "host=$(hostname) slot=$SLOT seed=$SEED start=$(date -Is)"

for F in 1 2 3 4 5; do
  METRICS="${OUT_DIR}/metrics/fold${F}_seed${SEED}.json"
  if [ -f "$METRICS" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$F seed=$SEED: $METRICS exists"
    continue
  fi
  LF="${LOG_DIR}/fold${F}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] training fold=$F seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.biotech_nbi_layer1_eval \
    --universe biotech_nbi --fold "$F" --seed "$SEED" \
    --output-dir-root "$OUT_DIR" \
    > "$LF" 2>&1 || {
      echo "FAILED fold=$F seed=$SEED ; see $LF"
      continue
    }
done

echo "DONE slot=$SLOT seed=$SEED $(date -Is)"
