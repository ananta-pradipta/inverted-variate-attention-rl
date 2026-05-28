#!/bin/bash
#SBATCH --job-name=tli-sp500-verify
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=sp500_panel_verify_%j.out

# Verify the independent S&P 500 panel loader against the real data
# artifacts on Wulver (no local compute). Reads only the produced parquet
# files under ~/phd-research/data; imports no other-project code.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/three_layer_invar}"
VENV="${VENV:-$HOME/three_layer_invar-env}"
cd "$REPO_DIR"

module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python -m pip install pyarrow==15.0.2 >/dev/null

export PYTHONPATH="$REPO_DIR"
export TLI_SP500_DATA_ROOT="$HOME/phd-research/data"

echo "S&P 500 panel verification on $(hostname)"
echo "data root: $TLI_SP500_DATA_ROOT"
python -m pytest -q tests/test_sp500_panel.py
echo "S&P 500 panel verification complete"
