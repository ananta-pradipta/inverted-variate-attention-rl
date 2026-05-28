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

# Phase B of the InVAR-RL baseline audit. Re-evaluate four Layer-1
# ranker baselines under their AUTHORS' native long-only top-K
# protocols on the universal lattice_native panel. Pairs with the
# existing top-K L/S wrapper eval in
# invar_rl/training/native_ranker_eval.py. CPU-only, ~30 sec / cell.
# 4 baselines x 5 folds x 5 seeds = 100 cells, plus rollup.

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

OUT_DIR="invar_rl/results/baselines_long_only"
LOG_DIR="invar_rl/logs/baselines_long_only"
NPZ_ROOT="results/baselines_universal_two_regime_val"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is)"

# Native K per baseline (Duan AAAI'22, Li AAAI'24, Fan+Shen AAAI'24, Gu ICAIF'24).
declare -A NATIVE_K=(
  [factorvae]=50
  [master]=30
  [stockmixer]=25
  [dystage]=25
)

for B in factorvae master stockmixer dystage; do
  K=${NATIVE_K[$B]}
  mkdir -p "${OUT_DIR}/${B}"
  for F in 1 2 3 4 5; do
    LF="${LOG_DIR}/${B}_fold${F}_sweep.log"
    # --sweep-fold builds the lattice bridge once and reuses it for all
    # 5 seeds (42..46) of this fold; the bridge is invariant across
    # seeds and was the per-cell bottleneck. Each cell still has its
    # own skip-if-output-exists guard.
    PYTHONPATH=$PWD python3 -u -m invar_rl.training.baselines_long_only_eval \
      --baseline "$B" --fold "$F" --sweep-fold --top-k "$K" \
      --npz-root "$NPZ_ROOT" \
      --output-dir-root "$OUT_DIR" \
      --panel_kind lattice_native \
      --two_regime_val \
      > "$LF" 2>&1
  done
  echo "[$(date +%H:%M:%S)] baseline=$B done"
done

echo "[$(date +%H:%M:%S)] running rollup"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.baselines_long_only_eval \
  --rollup \
  --output-dir-root "$OUT_DIR"

echo "DONE $(date -Is)"
