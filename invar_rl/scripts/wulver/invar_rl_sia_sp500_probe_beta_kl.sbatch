#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# InVAR-RL-SIA Phase 1 beta_kl probe: SP500 fold 1, seed 42.
# Brackets the Phase 1 default beta_kl=1e-3 with {1e-4, 1e-2}.
# Internal env-var sweep, two cells, sequential on one GPU.
#
# Phase 1 default beta_kl=1e-3 already lives at
#   outputs/sp500/layer2_sia/smoke/summary/fold1_seed42.json
# so this sbatch does NOT re-run it.
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sia_sp500_probe_beta_kl.sbatch

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
SEED=42

BETA_KL_LIST=("1e-4" "1e-2")
TAG_LIST=("beta_1e-4" "beta_1e-2")

LOG_ROOT="invar_rl/logs/sp500_layer2_sia_probe_beta_kl"
CKPT_ROOT="invar_rl/results/stage1/_ckpt"
mkdir -p "$LOG_ROOT"

echo "host=$(hostname) start=$(date -Is) fold=$FOLD seed=$SEED timesteps=$TIMESTEPS"
nvidia-smi | head -8 || true

CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
if [ ! -f "$CKPT" ]; then
  echo "missing layer1 ckpt: $CKPT"
  exit 1
fi

for i in 0 1; do
  BETA_KL="${BETA_KL_LIST[$i]}"
  TAG="${TAG_LIST[$i]}"
  OUT_ROOT="outputs/sp500/layer2_sia/probe_beta_kl/${TAG}"
  mkdir -p "$OUT_ROOT"
  LF="${LOG_ROOT}/fold${FOLD}_seed${SEED}_${TAG}.log"
  echo "[$(date +%H:%M:%S)] run SIA probe beta_kl=${BETA_KL} -> ${OUT_ROOT}"

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
        echo "FAILED SIA probe beta_kl=${BETA_KL}; see $LF"
        tail -50 "$LF"
        exit 3
      }
  echo "[$(date +%H:%M:%S)] DONE beta_kl=${BETA_KL}"
  ls -la "$OUT_ROOT/" "$OUT_ROOT/summary/" 2>/dev/null || true
done

echo "DONE probe $(date -Is)"
