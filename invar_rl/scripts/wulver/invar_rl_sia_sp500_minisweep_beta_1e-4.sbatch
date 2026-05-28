#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# InVAR-RL-SIA Phase 1 mini-sweep: SP500 fold 1, single seed per slot.
# 5 parallel sbatch submissions, one per seed in {42, 43, 44, 45, 46}.
# beta_kl=1e-4 (best of the 3-cell probe, user-approved option A).
# Phase 1 defaults: latent_dim=16, lambda_gate=1e-4, lambda_inv=0.1,
# total_timesteps=20000, group_source=macro_kmeans_8.
#
# Submit (one per seed):
#   for s in 42 43 44 45 46; do
#     SEED=$s sbatch invar_rl/scripts/wulver/invar_rl_sia_sp500_minisweep_beta_1e-4.sbatch
#   done

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

TIMESTEPS="${TIMESTEPS:-20000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"
FOLD=1
SEED="${SEED:?must set SEED env var, e.g. SEED=42}"
BETA_KL="${BETA_KL:-1e-4}"

OUT_ROOT="outputs/sp500/layer2_sia/minisweep_beta_1e-4"
LOG_ROOT="invar_rl/logs/sp500_layer2_sia_minisweep_beta_1e-4"
CKPT_ROOT="invar_rl/results/stage1/_ckpt"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "host=$(hostname) start=$(date -Is) fold=$FOLD seed=$SEED beta_kl=$BETA_KL timesteps=$TIMESTEPS"
nvidia-smi | head -8 || true

CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
if [ ! -f "$CKPT" ]; then
  echo "missing layer1 ckpt: $CKPT"
  exit 1
fi

LF="${LOG_ROOT}/fold${FOLD}_seed${SEED}.log"
echo "[$(date +%H:%M:%S)] run SIA mini-sweep fold=$FOLD seed=$SEED beta_kl=$BETA_KL"

PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_layer2_sia \
    --fold "$FOLD" --seed "$SEED" \
    --total-timesteps "$TIMESTEPS" \
    --eval-freq "$EVAL_FREQ" \
    --output-dir-root "$OUT_ROOT" \
    --layer1-ckpt-root "$CKPT_ROOT" \
    --layer3 invar_rl/configs/layer3.yaml \
    --stage3 invar_rl/configs/stage3.yaml \
    --latent-dim 16 --beta-kl "$BETA_KL" \
    --lambda-gate 1e-4 --lambda-inv 0.1 \
    --group-source macro_kmeans_8 \
    > "$LF" 2>&1 || {
      echo "FAILED SIA mini-sweep fold=$FOLD seed=$SEED beta_kl=$BETA_KL; see $LF"
      tail -50 "$LF"
      exit 3
    }

echo "DONE mini-sweep fold=$FOLD seed=$SEED beta_kl=$BETA_KL $(date -Is)"
ls -la "$OUT_ROOT/" "$OUT_ROOT/summary/" 2>/dev/null | head -20 || true
