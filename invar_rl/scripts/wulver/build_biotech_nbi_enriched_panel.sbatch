#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=general
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Build the biotech NBI ENRICHED 22-feature panel. CPU-only, ~10 min
# wall (price-feature loop dominates; StockTwits + fundamentals joins
# are fast on the NBI universe).
#
# Writes data/biotech_nbi/panel_features_enriched.parquet.

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

OUT_PARQUET="data/biotech_nbi/panel_features_enriched.parquet"
LOG_DIR="invar_rl/logs/biotech_nbi_enriched_panel_build"
mkdir -p "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is)"

if [ -f "$OUT_PARQUET" ]; then
  echo "[$(date +%H:%M:%S)] skip: $OUT_PARQUET already exists"
  exit 0
fi

LF="$LOG_DIR/build.log"
echo "[$(date +%H:%M:%S)] building enriched panel"
PYTHONPATH=$PWD python3 -u invar_rl/scripts/build_biotech_nbi_enriched_panel.py \
  --start 2014-09-01 --end 2025-12-31 \
  2>&1 | tee "$LF"

if [ ! -f "$OUT_PARQUET" ]; then
  echo "FAILED: $OUT_PARQUET not produced; see $LF"
  exit 1
fi

echo "DONE $(date -Is)"
