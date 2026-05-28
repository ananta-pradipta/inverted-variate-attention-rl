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

# InVAR-RL-SIA Phase 1 smoke: SP500 fold 1, seed 42 only.
# Single (fold, seed) cell on the canonical equal-weight K=50 wrapper with
# the SIA SAC variant. Confirms pipeline correctness before the Phase 2
# 5-cell fold-1 mini-sweep is requested. Phase 2 needs a separate,
# user-approved sbatch.
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sia_sp500_smoke.sbatch

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

OUT_ROOT="outputs/sp500/layer2_sia/smoke"
LOG_ROOT="invar_rl/logs/sp500_layer2_sia_smoke"
CKPT_ROOT="invar_rl/results/stage1/_ckpt"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "host=$(hostname) start=$(date -Is) fold=$FOLD seed=$SEED timesteps=$TIMESTEPS"
nvidia-smi | head -8 || true

CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
if [ ! -f "$CKPT" ]; then
  echo "missing layer1 ckpt: $CKPT"
  exit 1
fi

# Tests gate: must pass the 5 SIA unit tests before training begins.
echo "[$(date +%H:%M:%S)] running SIA unit tests"
PYTHONPATH=$PWD python3 -m pytest -x -q \
    tests/test_sia_actor.py \
    tests/test_sia_aux_loss.py \
    tests/test_sia_gate_learnability.py \
    tests/test_sia_critic.py \
    tests/test_sia_sac_subclass.py \
    > "$LOG_ROOT/tests.log" 2>&1 || {
      echo "FAILED SIA unit tests; see $LOG_ROOT/tests.log"
      tail -50 "$LOG_ROOT/tests.log"
      exit 2
    }
echo "tests pass"
tail -3 "$LOG_ROOT/tests.log"

LF="${LOG_ROOT}/fold${FOLD}_seed${SEED}.log"
echo "[$(date +%H:%M:%S)] run SIA smoke fold=$FOLD seed=$SEED"

PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_layer2_sia \
    --fold "$FOLD" --seed "$SEED" \
    --total-timesteps "$TIMESTEPS" \
    --eval-freq "$EVAL_FREQ" \
    --output-dir-root "$OUT_ROOT" \
    --layer1-ckpt-root "$CKPT_ROOT" \
    --layer3 invar_rl/configs/layer3.yaml \
    --stage3 invar_rl/configs/stage3.yaml \
    --latent-dim 16 --beta-kl 1e-3 \
    --lambda-gate 1e-4 --lambda-inv 0.1 \
    --group-source macro_kmeans_8 \
    > "$LF" 2>&1 || {
      echo "FAILED SIA smoke fold=$FOLD seed=$SEED; see $LF"
      tail -50 "$LF"
      exit 3
    }

echo "DONE smoke $(date -Is)"
ls -la "$OUT_ROOT/" "$OUT_ROOT/summary/" 2>/dev/null || true
