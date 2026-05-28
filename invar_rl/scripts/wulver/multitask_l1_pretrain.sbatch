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
#SBATCH --mem=64G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Option B Stage-1: multi-task contrastive pretrain of the canonical
# PerTickerTemporalEncoder backbone JOINTLY across SP500 + NDX + NBI.
# Single sbatch (per [[feedback_wulver_single_job]]) with an internal
# loop over the 5 folds; one seed (42) per the canonical Stage-1
# convention (Stage 1 is fold-keyed not seed-keyed; Stage 2 seeds reuse
# the same per-fold encoder).
#
# Generic job name (model-eval) per [[feedback_wulver_job_names]].
#
# Submit:
#   sbatch invar_rl/scripts/wulver/multitask_l1_pretrain.sbatch

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

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
PRETRAIN_DIR="invar_rl/results/multitask_l1"
LOG_DIR="invar_rl/logs/multitask_l1_pretrain"
mkdir -p "$PRETRAIN_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) epochs=${PRETRAIN_EPOCHS}"
nvidia-smi | head -8 || true

for FOLD in 1 2 3 4 5; do
  CKPT_SP="${PRETRAIN_DIR}/_ckpt_per_universe/lattice_native/fold${FOLD}_encoder.pt"
  CKPT_NDX="${PRETRAIN_DIR}/_ckpt_per_universe/nasdaq100/fold${FOLD}_encoder.pt"
  CKPT_NBI="${PRETRAIN_DIR}/_ckpt_per_universe/biotech_nbi_enriched/fold${FOLD}_encoder.pt"
  if [ -f "$CKPT_SP" ] && [ -f "$CKPT_NDX" ] && [ -f "$CKPT_NBI" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD: all 3 universe ckpts present"
    continue
  fi
  LF="${LOG_DIR}/fold${FOLD}.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD"
  PYTHONPATH=$PWD python3 -u -m src.baselines.train_multitask_pretrain \
    --fold "$FOLD" --seed 42 \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --output_dir "$PRETRAIN_DIR" \
    > "$LF" 2>&1 || {
      echo "FAILED fold=$FOLD; see $LF"
      tail -40 "$LF"
      continue
    }
done

echo "DONE $(date -Is)"
ls -la "$PRETRAIN_DIR/_ckpt_per_universe/" 2>/dev/null || true
