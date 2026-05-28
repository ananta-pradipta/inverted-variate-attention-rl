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

# InVAR-RL-SIA Phase 5 L/O: NBI-enriched 25-cell sweep, long-only protocol.
# Full SIA at beta_kl=1e-4, lambda_inv=0.1, K=25 long-only top-K.
#
# Submit:
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_sia_biotech_nbi_enriched_phase5_lo.sbatch
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
OUT_ROOT="outputs/biotech_nbi_enriched/layer2_sia/phase5_lo"
LOG_ROOT="invar_rl/logs/biotech_nbi_enriched_layer2_sia_phase5_lo"
CKPT_ROOT="outputs/biotech_nbi_enriched/layer1/_ckpt"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "host=$(hostname) start=$(date -Is) slot=$SLOT fold=$FOLD long_only=ON"
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

  LF="${LOG_ROOT}/phase5_lo_fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] phase5_lo fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.biotech_nbi_enriched_layer2_sia \
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
      --universe-label biotech_nbi_enriched \
      --long-only \
      > "$LF" 2>&1 || {
        echo "FAILED fold=$FOLD seed=$SEED; see $LF"
        tail -50 "$LF"
        exit 6
      }
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f'  fold=${FOLD} seed=${SEED} test_sharpe={d[\"test_pooled_sharpe\"]:+.4f}')" "$SUM"
done

echo "[$(date +%H:%M:%S)] slot=$SLOT fold=$FOLD done"
ls -la "$OUT_ROOT/summary/" 2>/dev/null | grep "fold${FOLD}_" | head -20 || true
