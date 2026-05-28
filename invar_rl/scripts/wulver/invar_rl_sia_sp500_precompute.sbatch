#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=general
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Precompute the SIA regime-probs cache (KMeans-8 fit + per-day softmax)
# for SP500 fold 1. CPU-only; ~5 min wall.
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sia_sp500_precompute.sbatch
#
# Produces:
#   cache/dr_rl/regime_probs/sp500/fold1/kmeans_fit.npz
#   cache/dr_rl/regime_probs/sp500/fold1/probs.parquet

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

FOLD=1
UNIVERSE=sp500

echo "host=$(hostname) start=$(date -Is) precompute regime_probs universe=$UNIVERSE fold=$FOLD"

PYTHONPATH=$PWD python3 -u -c "
from pathlib import Path
from src.invar import InVARConfig
from invar_rl.data.lattice_bridge import build_lattice_bridge
from invar_rl.layer2_sia.regime_probs import precompute_all

cfg = InVARConfig(fold=${FOLD}, seed=42)
cfg.panel_kind = 'lattice_native'
cfg.two_regime_val = True
cfg.panel_end = '2025-12-31'
cfg.enable_retrieval_bank = False
bridge = build_lattice_bridge(cfg)
print(f'bridge built: n_dates={len(bridge.dates)} macro_dim={bridge.macro_arr.shape[1]}')
print(f'train_idx={len(bridge.train_idx)} val_idx={len(bridge.val_idx)} test_idx={len(bridge.test_idx)}')

out = precompute_all(
    universe='${UNIVERSE}',
    fold=${FOLD},
    bridge=bridge,
    temperature=1.0,
)
print(f'wrote fit={out[\"fit_path\"]}')
print(f'wrote probs={out[\"probs_path\"]} (n_days={out[\"n_days\"]})')
"

echo "DONE precompute $(date -Is)"
ls -la "cache/dr_rl/regime_probs/${UNIVERSE}/fold${FOLD}/"
