#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# InVAR-RL-SIA Phase 2B: lambda_inv mini-sweep on SP500 fold 1.
# One slot per SEED; LAMBDA_INV env var controls the regularisation weight.
# Output dir: outputs/sp500/layer2_sia/minisweep_lambda_inv_{LAMBDA_INV}/
#
# Submit (per lambda_inv value, 5 parallel slots = 5 seeds):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT,LAMBDA_INV=0.3 \
#       invar_rl/scripts/wulver/invar_rl_sia_sp500_lambda_minisweep.sbatch
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
SEEDS=(42 43 44 45 46)
if [ "$SLOT" -lt 0 ] || [ "$SLOT" -ge 5 ]; then
  echo "invalid SLOT=$SLOT; must be 0..4"
  exit 1
fi
SEED="${SEEDS[$SLOT]}"
FOLD=1

TIMESTEPS="${TIMESTEPS:-20000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"
BETA_KL="${BETA_KL:-1e-4}"
LAMBDA_INV="${LAMBDA_INV:-0.3}"
LAMBDA_GATE="${LAMBDA_GATE:-1e-4}"
LATENT_DIM="${LATENT_DIM:-16}"

OUT_ROOT="outputs/sp500/layer2_sia/minisweep_lambda_inv_${LAMBDA_INV}"
LOG_ROOT="invar_rl/logs/sp500_layer2_sia_lambda_minisweep"
CKPT_ROOT="invar_rl/results/stage1/_ckpt"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "host=$(hostname) start=$(date -Is) slot=$SLOT seed=$SEED fold=$FOLD lambda_inv=$LAMBDA_INV beta_kl=$BETA_KL"
nvidia-smi | head -8 || true

CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
if [ ! -f "$CKPT" ]; then
  echo "missing layer1 ckpt: $CKPT"
  exit 5
fi

SUM="${OUT_ROOT}/summary/fold${FOLD}_seed${SEED}.json"
if [ -f "$SUM" ]; then
  echo "[$(date +%H:%M:%S)] skip seed=$SEED (summary exists)"
  exit 0
fi

LF="${LOG_ROOT}/lambda_${LAMBDA_INV}_fold${FOLD}_seed${SEED}.log"
echo "[$(date +%H:%M:%S)] minisweep fold=$FOLD seed=$SEED beta_kl=$BETA_KL lambda_inv=$LAMBDA_INV regime_label=ON"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_layer2_sia \
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
    --universe-label sp500 \
    > "$LF" 2>&1 || {
      echo "FAILED fold=$FOLD seed=$SEED lambda_inv=$LAMBDA_INV; see $LF"
      tail -50 "$LF"
      exit 6
    }
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); ts=d.get('sia_train_stats',{}); print(f'  lambda=${LAMBDA_INV} seed=${SEED} test_sharpe={d[\"test_pooled_sharpe\"]:+.4f} aux_inv={ts.get(\"aux_inv\",0.0):.4e} gate_open={ts.get(\"gate_open_fraction\",0.0):+.4f}')" "$SUM"

echo "[$(date +%H:%M:%S)] slot=$SLOT seed=$SEED lambda_inv=$LAMBDA_INV done"
