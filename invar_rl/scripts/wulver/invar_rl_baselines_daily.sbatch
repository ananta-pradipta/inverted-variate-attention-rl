#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Re-run the two whole-stack RL baselines (FinRL A2C, StockFormer) on the
# universal S&P 500 long-only protocol, 5 folds x 5 seeds, with the patched
# evaluators that also dump the per-day return series (perf["daily_log_returns"]).
# Writes to fresh *_daily dirs so the published summary JSONs are untouched;
# the per-fold summaries should reproduce the headline numbers (same seeds/config).
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

FIN_OUT="invar_rl/results/finrl_faithful_daily"
SF_OUT="invar_rl/results/stockformer_faithful_daily"
LOG="invar_rl/logs/baselines_daily"
mkdir -p "$FIN_OUT" "$SF_OUT" "$LOG"
echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -5 || true

for FOLD in 1 2 3 4 5; do
  for SEED in 42 43 44 45 46; do
    echo "=== FinRL A2C fold$FOLD seed$SEED $(date -Is) ==="
    PYTHONPATH=$PWD python3 -u -m invar_rl.training.finrl_faithful_eval \
      --phase universal --method a2c --fold "$FOLD" --seed "$SEED" \
      --output-dir-root "$FIN_OUT" --total-timesteps 50000 \
      > "$LOG/finrl_a2c_f${FOLD}_s${SEED}.log" 2>&1 || echo "FINRL FAIL f$FOLD s$SEED"
    echo "=== StockFormer fold$FOLD seed$SEED $(date -Is) ==="
    PYTHONPATH=$PWD python3 -u -m invar_rl.training.stockformer_faithful_eval \
      --phase universal --fold "$FOLD" --seed "$SEED" \
      --output-dir-root "$SF_OUT" --total-timesteps 30000 --pretrain-epochs 30 \
      > "$LOG/stockformer_f${FOLD}_s${SEED}.log" 2>&1 || echo "SF FAIL f$FOLD s$SEED"
  done
done
echo "done=$(date -Is)"
