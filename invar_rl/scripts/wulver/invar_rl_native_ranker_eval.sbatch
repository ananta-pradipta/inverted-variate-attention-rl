#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=general
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=02:00:00
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

# Native top-K L/S evaluation for all 7 Layer-1 ranker baselines, 5
# folds, 5 seeds = 175 cells. CPU-only, no RL training. ~2 min per
# baseline; one job runs all of them.
OUT_DIR="invar_rl/results/native_ranker_baselines"
LOG_DIR="invar_rl/logs/native_ranker_baselines"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is)"

for B in master factorvae itransformer stockmixer dystage mera swa_invar; do
  for F in 1 2 3 4 5; do
    for S in 42 43 44 45 46; do
      OUT="${OUT_DIR}/${B}/fold${F}_seed${S}.json"
      if [ -f "$OUT" ]; then continue; fi
      LF="${LOG_DIR}/${B}_fold${F}_seed${S}.log"
      mkdir -p "${OUT_DIR}/${B}" "${LOG_DIR}"
      PYTHONPATH=$PWD python3 -u -m invar_rl.training.native_ranker_eval \
        --baseline "$B" --fold "$F" --seed "$S" \
        --output-dir-root "$OUT_DIR" \
        --panel_kind lattice_native \
        --two_regime_val \
        > "$LF" 2>&1
    done
  done
  echo "[$(date +%H:%M:%S)] baseline=$B done"
done

echo "DONE $(date -Is)"
