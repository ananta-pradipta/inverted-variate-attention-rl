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

# InVAR-RL-SIA Phase 3: NASDAQ-100 25-cell sweep (5 folds x 5 seeds).
# Regime-wired SIA at beta_kl=1e-4, lambda_inv=$LAMBDA_INV (winner from
# Phase 2B), wrapper K=20 per side.
#
# Per-slot worker: SLOT selects the fold (0..4 -> folds 1..5); each slot
# loops over 5 seeds (42..46) sequentially. 5 slots run in parallel for
# total wall ~40-50 min.
#
# Submit:
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT,LAMBDA_INV=<X> \
#       invar_rl/scripts/wulver/invar_rl_sia_nasdaq100_phase3.sbatch
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

SLOT="${SLOT:-0}"
FOLD=$((SLOT + 1))
if [ "$FOLD" -lt 1 ] || [ "$FOLD" -gt 5 ]; then
  echo "invalid SLOT=$SLOT; must be 0..4"
  exit 1
fi

TIMESTEPS="${TIMESTEPS:-20000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"
BETA_KL="${BETA_KL:-1e-4}"
LAMBDA_INV="${LAMBDA_INV:-0.1}"
LAMBDA_GATE="${LAMBDA_GATE:-1e-4}"
LATENT_DIM="${LATENT_DIM:-16}"
SEEDS=(42 43 44 45 46)

OUT_ROOT="outputs/nasdaq100/layer2_sia/phase3"
LOG_ROOT="invar_rl/logs/nasdaq100_layer2_sia_phase3"
CKPT_ROOT="outputs/nasdaq100/layer1/_ckpt"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "host=$(hostname) start=$(date -Is) slot=$SLOT fold=$FOLD lambda_inv=$LAMBDA_INV"
nvidia-smi | head -8 || true

for SEED in "${SEEDS[@]}"; do
  CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
  if [ ! -f "$CKPT" ]; then
    echo "missing layer1 ckpt: $CKPT"
    exit 5
  fi

  SUM="${OUT_ROOT}/summary/fold${FOLD}_seed${SEED}.json"
  if [ -f "$SUM" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED (summary exists)"
    continue
  fi

  LF="${LOG_ROOT}/phase3_fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] phase3 fold=$FOLD seed=$SEED lambda_inv=$LAMBDA_INV regime_label=ON"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.nasdaq100_layer2_sia \
      --fold "$FOLD" --seed "$SEED" \
      --total-timesteps "$TIMESTEPS" \
      --eval-freq "$EVAL_FREQ" \
      --output-dir-root "$OUT_ROOT" \
      --layer1-ckpt-root "$CKPT_ROOT" \
      --layer3 invar_rl/configs/layer3.yaml \
      --stage3 invar_rl/configs/stage3.yaml \
      --latent-dim "$LATENT_DIM" --beta-kl "$BETA_KL" \
      --lambda-gate "$LAMBDA_GATE" --lambda-inv "$LAMBDA_INV" \
      --group-source macro_kmeans_8 \
      --regime-label \
      --universe-label nasdaq100 \
      > "$LF" 2>&1 || {
        echo "FAILED fold=$FOLD seed=$SEED; see $LF"
        tail -50 "$LF"
        exit 6
      }
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); ts=d.get('sia_train_stats',{}); print(f'  fold=${FOLD} seed=${SEED} test_sharpe={d[\"test_pooled_sharpe\"]:+.4f} aux_inv={ts.get(\"aux_inv\",0.0):.4e}')" "$SUM"
done

echo "[$(date +%H:%M:%S)] slot=$SLOT fold=$FOLD done"
ls -la "$OUT_ROOT/summary/" 2>/dev/null | grep "fold${FOLD}_" | head -20 || true
