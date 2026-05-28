#!/bin/bash
#SBATCH --job-name=tli-phase4-verify
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=phase4_verify_%j.out

# Phase 4 verification on Wulver (no local compute): add the gymnasium
# dependency to the dedicated project venv, then run the full pytest suite,
# which includes the Layer 3 environment tests (gymnasium API conformance,
# constant and random sanity policies, differential-Sharpe verification,
# and the no-future-information observation check). No RL agent and no
# training in this phase.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/three_layer_invar}"
VENV="${VENV:-$HOME/three_layer_invar-env}"
cd "$REPO_DIR"

module load python3 2>/dev/null || true
source "$VENV/bin/activate"

python -m pip install gymnasium==0.29.1

export PYTHONPATH="$REPO_DIR"
echo "Phase 4 verification starting on $(hostname)"
python -m pytest -q
echo "Phase 4 verification complete"
