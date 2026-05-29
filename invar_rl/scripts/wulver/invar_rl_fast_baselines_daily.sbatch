#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Fast daily-return baselines for Figure 8: the three quick rankers
# (MASTER, FactorVAE, StockMixer) re-trained to regenerate prediction npz then
# evaluated under both books with per-day series, plus FinRL A2C. DySTAGE and
# StockFormer are excluded here (DySTAGE rebuilds a ~3h graph cache per cell;
# StockFormer SAC ~3h per cell), so they would not finish in walltime and are
# handled separately.
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
REPO="${REPO:-$HOME/phd-research}"; VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"; module load bright 2>/dev/null || true; module load python3 2>/dev/null || true
source "$VENV/bin/activate"

NPZ="results/baselines_universal_two_regime_val_daily"
LO="invar_rl/results/baselines_long_only_daily"
LS="invar_rl/results/native_ranker_baselines_daily"
FIN="invar_rl/results/finrl_faithful_daily"
LOG="invar_rl/logs/fast_baselines_daily"; mkdir -p "$LOG"
echo "host=$(hostname) start=$(date -Is)"; nvidia-smi | head -5 || true
declare -A TOPK=( [master]=30 [factorvae]=50 [stockmixer]=25 )

# Fast rankers: train -> eval both books, all cells.
for FOLD in 1 2 3 4 5; do for SEED in 42 43 44 45 46; do for NAME in master factorvae stockmixer; do
  echo "=== train ${NAME} f${FOLD} s${SEED} $(date -Is) ==="
  PYTHONPATH=$PWD python3 -u -m "src.baselines.train_${NAME}_v2" --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native --two_regime_val --output_dir "${NPZ}/${NAME}" \
    > "$LOG/train_${NAME}_f${FOLD}_s${SEED}.log" 2>&1 || { echo "TRAIN FAIL ${NAME} f$FOLD s$SEED"; continue; }
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.baselines_long_only_eval --baseline "$NAME" \
    --fold "$FOLD" --seed "$SEED" --top-k "${TOPK[$NAME]}" --npz-root "$NPZ" --output-dir-root "$LO" \
    > "$LOG/lo_${NAME}_f${FOLD}_s${SEED}.log" 2>&1 || echo "LO FAIL ${NAME} f$FOLD s$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.native_ranker_eval --baseline "$NAME" \
    --fold "$FOLD" --seed "$SEED" --npz-root "$NPZ" --output-dir-root "$LS" \
    > "$LOG/ls_${NAME}_f${FOLD}_s${SEED}.log" 2>&1 || echo "LS FAIL ${NAME} f$FOLD s$SEED"
done; done; done

# FinRL A2C: fast whole-stack RL, daily series.
for FOLD in 1 2 3 4 5; do for SEED in 42 43 44 45 46; do
  echo "=== FinRL A2C f${FOLD} s${SEED} $(date -Is) ==="
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.finrl_faithful_eval --phase universal --method a2c \
    --fold "$FOLD" --seed "$SEED" --output-dir-root "$FIN" --total-timesteps 50000 \
    > "$LOG/finrl_f${FOLD}_s${SEED}.log" 2>&1 || echo "FINRL FAIL f$FOLD s$SEED"
done; done
echo "done=$(date -Is)"
