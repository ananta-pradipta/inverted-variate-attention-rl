#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# InVAR-RL-Q Phase 1 smoke: SP500 fold 1, seed 42 only.
# Single (fold, seed) cell on the canonical equal-weight K=50 wrapper with
# the Q SAC variant (twin QuantileCritic + CVaR-blend actor objective).
# Confirms pipeline correctness before the Phase 2 25-cell sweep is
# requested. Phase 2 needs a separate, user-approved sbatch.
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_q_sp500_phase1_smoke.sbatch

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
N_QUANTILES="${N_QUANTILES:-51}"
ALPHA_CVAR="${ALPHA_CVAR:-0.1}"
ETA_BLEND="${ETA_BLEND:-0.5}"
FOLD=1
SEED=42

OUT_ROOT="outputs/sp500/layer2_q/phase1_smoke"
LOG_ROOT="invar_rl/logs/sp500_layer2_q_phase1_smoke"
CKPT_ROOT="invar_rl/results/stage1/_ckpt"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "host=$(hostname) start=$(date -Is) fold=$FOLD seed=$SEED timesteps=$TIMESTEPS"
nvidia-smi | head -8 || true

CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
if [ ! -f "$CKPT" ]; then
  echo "missing layer1 ckpt: $CKPT"
  exit 1
fi

# Single-cell internal loop (one iteration); structure matches the SIA
# smoke sbatch so a future Phase 2 sweep slots in via for-loop expansion.
for SEED_ITER in "$SEED"; do
  LF="${LOG_ROOT}/fold${FOLD}_seed${SEED_ITER}.log"
  echo "[$(date +%H:%M:%S)] run Q smoke fold=$FOLD seed=$SEED_ITER"

  PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_layer2_q \
      --fold "$FOLD" --seed "$SEED_ITER" \
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
        echo "FAILED Q smoke fold=$FOLD seed=$SEED_ITER; see $LF"
        tail -50 "$LF"
        exit 3
      }
done

echo "DONE smoke $(date -Is)"
ls -la "$OUT_ROOT/" "$OUT_ROOT/summary/" 2>/dev/null || true
