#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=general
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Path B data rebuild: extend SP500 raw inputs through 2025-12-31, then
# rebuild the lattice panel for 2755 trading days. Single CPU sbatch
# (general partition) per [[feedback_wulver_single_job]]; generic job
# name per [[feedback_wulver_job_names]]. Outputs land at:
#   data/raw/sp500/sp500_constituents_history_extended.parquet
#   data/raw/sp500/prices_sp500_extended.parquet
#   data/lattice/raw/{macro_fred*.csv, macro_etfs_extra.parquet} (re-pulled)
#   data/processed/risk_features_sp500_ext.parquet
#   data/lattice/raw/{prices_sp500,sp500_constituents_pit,fundamentals_edgar_sp500,
#                     macro_etfs,macro_fred}.{parquet,csv}  (symlinks)
#   data/lattice/processed/panel_features.parquet (2755 days)
#   data/lattice/processed/active_mask.parquet
#   data/lattice/processed/cohorts.parquet
#   data/lattice/processed/macro_state.parquet

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

LOG_DIR="invar_rl/logs/sp500_panel_rebuild"
mkdir -p "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is)"

# --------------------------------------------------------------------
# Step 1: Extend SP500 constituents PIT history through current date.
# Pulls Wikipedia change-log and SEC EDGAR for new tickers.
# Output: data/raw/sp500/sp500_constituents_history_extended.parquet
# --------------------------------------------------------------------
EXT_CONS="data/raw/sp500/sp500_constituents_history_extended.parquet"
if [ -f "$EXT_CONS" ]; then
  echo "[$(date +%H:%M:%S)] step 1 skip (exists: $EXT_CONS)"
else
  echo "[$(date +%H:%M:%S)] step 1: extend_constituents_pit.py"
  PYTHONPATH=$PWD python3 -u scripts/lattice/extend_constituents_pit.py \
    > "$LOG_DIR/step1_extend_constituents.log" 2>&1
fi

# --------------------------------------------------------------------
# Step 2: Extend SP500 prices through 2025-12-31 via yfinance.
# Output: data/raw/sp500/prices_sp500_extended.parquet
# --------------------------------------------------------------------
EXT_PRICES="data/raw/sp500/prices_sp500_extended.parquet"
if [ -f "$EXT_PRICES" ]; then
  echo "[$(date +%H:%M:%S)] step 2 skip (exists: $EXT_PRICES)"
else
  echo "[$(date +%H:%M:%S)] step 2: extend_prices.py"
  PYTHONPATH=$PWD python3 -u scripts/lattice/extend_prices.py \
    > "$LOG_DIR/step2_extend_prices.log" 2>&1
fi

# --------------------------------------------------------------------
# Step 3: Extend macro raw (FRED, sector ETFs, broad ETFs) to 2025-12-31.
# Outputs (in-place updates):
#   data/raw/macro_fred_full.csv      (FRED full series including VIXCLS)
#   data/lattice/raw/macro_fred_extra.csv  (T10YIE, DTWEXBGS)
#   data/raw/sp500/sector_etfs.parquet (SPY, QQQ, XL*)
#   data/lattice/raw/macro_etfs_extra.parquet (IWM, HYG, TLT, GLD)
# Always re-run: this script does a full re-pull which is the safe path
# to fill any missing VIXCLS column.
# --------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] step 3: extend_macro_2025.py"
PYTHONPATH=$PWD python3 -u scripts/lattice/extend_macro_2025.py \
  > "$LOG_DIR/step3_extend_macro.log" 2>&1

# --------------------------------------------------------------------
# Step 4: Build extended risk_features (needed by build_macro_state).
# Output: data/processed/risk_features_sp500_ext.parquet
# We then overwrite the lattice macro builder's risk path to this one
# via a symlink so build_macro reads it.
# --------------------------------------------------------------------
EXT_RISK="data/processed/risk_features_sp500_ext.parquet"
if [ -f "$EXT_RISK" ]; then
  echo "[$(date +%H:%M:%S)] step 4 skip (exists: $EXT_RISK)"
else
  echo "[$(date +%H:%M:%S)] step 4: build_v2_aux_extended (risk only)"
  PYTHONPATH=$PWD python3 -u -c "
from scripts.lattice.build_v2_aux_extended import build_extended_risk_features
build_extended_risk_features()
" > "$LOG_DIR/step4_risk_features.log" 2>&1
fi

# --------------------------------------------------------------------
# Step 5: Materialise the lattice/raw/ symlinks so build_phase1 sees
# the extended raw inputs. The lattice builders expect canonical names
# in data/lattice/raw/.
# --------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] step 5: symlinks into data/lattice/raw"
mkdir -p data/lattice/raw

# Prices: extended file -> canonical lattice raw name.
ln -sf "$PWD/data/raw/sp500/prices_sp500_extended.parquet" \
       "data/lattice/raw/prices_sp500.parquet"
# Constituents PIT: extended file -> canonical lattice raw name.
ln -sf "$PWD/data/raw/sp500/sp500_constituents_history_extended.parquet" \
       "data/lattice/raw/sp500_constituents_pit.parquet"
# Fundamentals: existing extended sp500 fundamentals parquet (already to 2026-05).
ln -sf "$PWD/data/raw/sp500/fundamentals_sp500.parquet" \
       "data/lattice/raw/fundamentals_edgar_sp500.parquet"
# Macro ETFs (main): sector_etfs.parquet -> macro_etfs.parquet
ln -sf "$PWD/data/raw/sp500/sector_etfs.parquet" \
       "data/lattice/raw/macro_etfs.parquet"
# Macro FRED (main): macro_fred_full.csv -> macro_fred.csv (uses DGS3MO,
# DGS2, DGS10, BAA10Y, VIXCLS, VIXMCLS).
ln -sf "$PWD/data/raw/macro_fred_full.csv" \
       "data/lattice/raw/macro_fred.csv"
# Risk features: extended one for build_macro_state.
ln -sf "$PWD/data/processed/risk_features_sp500_ext.parquet" \
       "data/processed/risk_features_lattice.parquet"

ls -la data/lattice/raw/

# --------------------------------------------------------------------
# Step 6: Rebuild lattice panel + cohorts + active_mask + macro_state
# with panel_end=2025-12-31. Mirrors scripts/wulver/rebuild_panel_2025.sbatch
# (commit f8a575d era pattern) but inlined here so the full Path B is one
# atomic CPU job.
# --------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] step 6: build_panel_2025.py (build_phase1 + build_macro_state)"
# Patch build_macro_state to use the extended risk_features path via env var
# fallback (the script defaults to data/processed/risk_features.parquet which
# is the biotech one and only goes to 2022-12). We swap by overwriting that
# default for this run only.
PYTHONPATH=$PWD python3 -u <<'PYEOF' > "$LOG_DIR/step6_build_panel.log" 2>&1
import pandas as pd
from pathlib import Path
from src.lattice.data.build_panel import LatticePhase1Config, build_phase1
from src.lattice.data.build_macro import build_macro_state

print("=== build_phase1 panel_end=2025-12-31 ===", flush=True)
cfg = LatticePhase1Config(
    raw_dir=Path("data/lattice/raw"),
    out_dir=Path("data/lattice/processed"),
    panel_start="2015-01-09",
    panel_end="2025-12-31",
)
summary = build_phase1(cfg)
print(f"build_phase1 summary: {summary}", flush=True)

print("=== build_macro_state panel_end=2025-12-31 ===", flush=True)
# Use the extended risk-features parquet (covers 2014-09 -> 2026-04).
out = build_macro_state(
    risk_features_parquet=Path("data/processed/risk_features_sp500_ext.parquet"),
    panel_start="2015-01-09",
    panel_end="2025-12-31",
)
print(f"build_macro_state out: {out}", flush=True)
PYEOF

# --------------------------------------------------------------------
# Step 7: Verify extents.
# --------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] step 7: verify"
PYTHONPATH=$PWD python3 -u <<'PYEOF' > "$LOG_DIR/step7_verify.log" 2>&1
import pandas as pd
p = pd.read_parquet("data/lattice/processed/panel_features.parquet")
m = pd.read_parquet("data/lattice/processed/macro_state.parquet")
print(f"panel_features: rows={len(p):,} tickers={p['ticker'].nunique()} "
      f"days={p['date'].nunique()} "
      f"date_range=[{p['date'].min().date()}, {p['date'].max().date()}]",
      flush=True)
print(f"macro_state:    rows={len(m):,} cols={len(m.columns)} "
      f"date_range=[{m['date'].min().date()}, {m['date'].max().date()}]",
      flush=True)
print(f"panel cols ({len(p.columns)}): {p.columns.tolist()}", flush=True)
assert p['date'].nunique() >= 2700, (
    f"panel only has {p['date'].nunique()} days; expected ~2755"
)
print("VERIFY PASSED: panel has >= 2700 days", flush=True)
PYEOF
cat "$LOG_DIR/step7_verify.log"

echo "DONE $(date -Is)"
