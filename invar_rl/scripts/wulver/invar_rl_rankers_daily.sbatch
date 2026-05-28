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

# Re-train the four universal ranker baselines (MASTER, FactorVAE, StockMixer,
# DySTAGE) on the S&P 500 lattice_native panel + two-regime-val protocol to
# REGENERATE the prediction npz (the originals were deleted), then evaluate
# each cell under both books with the patched evaluators that now persist the
# per-day return series (methods[*]["daily_log_returns"]) needed for Figure 8.
#
# Long-only top-K per the published protocol: MASTER 30, FactorVAE 50,
# StockMixer 25, DySTAGE 25. Long-short uses topk_ls_k25.
#
# Writes to fresh *_daily dirs so the checkpointed published JSONs are
# untouched. Re-trained numbers will not be bit-identical to the published
# cells (GPU/library drift); the fresh numbers are adopted into the tables so
# the daily curve and the tables agree.
#
# 4 baselines x 5 folds x 5 seeds = 100 train+eval cells. Walltime 24h.
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

NPZ_ROOT="results/baselines_universal_two_regime_val_daily"
LO_OUT="invar_rl/results/baselines_long_only_daily"
LS_OUT="invar_rl/results/native_ranker_baselines_daily"
LOG="invar_rl/logs/rankers_daily"
mkdir -p "$LOG"
echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -5 || true

# Long-only top-K per published baseline protocol.
declare -A TOPK=( [master]=30 [factorvae]=50 [stockmixer]=25 [dystage]=25 )

for FOLD in 1 2 3 4 5; do
  for SEED in 42 43 44 45 46; do
    for NAME in master factorvae stockmixer dystage; do
      echo "=== train ${NAME} fold${FOLD} seed${SEED} $(date -Is) ==="
      PYTHONPATH=$PWD python3 -u -m "src.baselines.train_${NAME}_v2" \
        --fold "$FOLD" --seed "$SEED" \
        --panel_kind lattice_native --two_regime_val \
        --output_dir "${NPZ_ROOT}/${NAME}" \
        > "$LOG/train_${NAME}_f${FOLD}_s${SEED}.log" 2>&1 \
        || { echo "TRAIN FAIL ${NAME} f$FOLD s$SEED"; continue; }

      echo "=== eval long-only ${NAME} (top-${TOPK[$NAME]}) ==="
      PYTHONPATH=$PWD python3 -u -m invar_rl.training.baselines_long_only_eval \
        --baseline "$NAME" --fold "$FOLD" --seed "$SEED" \
        --top-k "${TOPK[$NAME]}" \
        --npz-root "$NPZ_ROOT" --output-dir-root "$LO_OUT" \
        > "$LOG/lo_${NAME}_f${FOLD}_s${SEED}.log" 2>&1 \
        || echo "LO EVAL FAIL ${NAME} f$FOLD s$SEED"

      echo "=== eval long-short ${NAME} (topk_ls_k25) ==="
      PYTHONPATH=$PWD python3 -u -m invar_rl.training.native_ranker_eval \
        --baseline "$NAME" --fold "$FOLD" --seed "$SEED" \
        --npz-root "$NPZ_ROOT" --output-dir-root "$LS_OUT" \
        > "$LOG/ls_${NAME}_f${FOLD}_s${SEED}.log" 2>&1 \
        || echo "LS EVAL FAIL ${NAME} f$FOLD s$SEED"
    done
  done
done
echo "done=$(date -Is)"
