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

# InVAR-RL-Q Phase 2: SP500 25-cell sweep, full SIA-style protocol.
# Twin QuantileCritic (51 quantiles) + CVaR-blend actor objective
# (eta=0.5, alpha=0.1). Wrapper K=50 per side (L/S protocol).
#
# Per-slot worker: SLOT selects the fold (0..4 -> folds 1..5); each
# slot loops over 5 seeds (42..46) sequentially. 5 slots in parallel.
#
# Submit:
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/invar_rl_q_sp500_phase2.sbatch
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
N_QUANTILES="${N_QUANTILES:-51}"
ALPHA_CVAR="${ALPHA_CVAR:-0.1}"
ETA_BLEND="${ETA_BLEND:-0.5}"
SEEDS=(42 43 44 45 46)
OUT_ROOT="outputs/sp500/layer2_q/phase2"
LOG_ROOT="invar_rl/logs/sp500_layer2_q_phase2"
CKPT_ROOT="invar_rl/results/stage1/_ckpt"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "host=$(hostname) start=$(date -Is) slot=$SLOT fold=$FOLD n_q=$N_QUANTILES alpha=$ALPHA_CVAR eta=$ETA_BLEND"
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

  LF="${LOG_ROOT}/phase2_fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] phase2 fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_layer2_q \
      --fold "$FOLD" --seed "$SEED" \
      --total-timesteps "$TIMESTEPS" \
      --eval-freq "$EVAL_FREQ" \
      --output-dir-root "$OUT_ROOT" \
      --layer1-ckpt-root "$CKPT_ROOT" \
      --layer3 invar_rl/configs/layer3.yaml \
      --stage3 invar_rl/configs/stage3.yaml \
      --n-quantiles "$N_QUANTILES" \
      --alpha-cvar "$ALPHA_CVAR" \
      --eta-blend "$ETA_BLEND" \
      > "$LF" 2>&1 || {
        echo "FAILED fold=$FOLD seed=$SEED; see $LF"
        tail -50 "$LF"
        exit 6
      }
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f'  fold=${FOLD} seed=${SEED} test_sharpe={d[\"test_pooled_sharpe\"]:+.4f}')" "$SUM"
done

echo "[$(date +%H:%M:%S)] slot=$SLOT fold=$FOLD done"
ls -la "$OUT_ROOT/summary/" 2>/dev/null | grep "fold${FOLD}_" | head -20 || true
