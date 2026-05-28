#!/bin/bash
#SBATCH --job-name=tli-phase2-verify
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=phase2_verify_%j.out

# Phase 2 verification: provision the dedicated project venv (independent of
# the RAG-STAR phd-research env), then run the full pytest suite, which
# includes the Layer 2 constraint, gradient-flow, and finite-difference
# checks. CPU only; the QP solves do not need a GPU.
#
# Only the Phase 0 to 2 dependency subset is installed here, at the pins in
# requirements.txt. The RL extras (gymnasium, stable-baselines3,
# sb3-contrib) are installed later when Phase 4 and 5 need them, to avoid a
# premature heavy and resolver-fragile install.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/three_layer_invar}"
VENV="${VENV:-$HOME/three_layer_invar-env}"
cd "$REPO_DIR"

module load python3 2>/dev/null || true

if [ ! -d "$VENV" ]; then
  echo "Creating project venv at $VENV"
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

python -m pip install --upgrade pip wheel >/dev/null
python -m pip install \
  torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install \
  numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2 PyYAML==6.0.1 \
  pytest==8.2.0 cvxpy==1.4.3 cvxpylayers==0.1.6

export PYTHONPATH="$REPO_DIR"
echo "Phase 2 verification starting on $(hostname)"
python -m pytest -q
echo "Phase 2 verification complete"
