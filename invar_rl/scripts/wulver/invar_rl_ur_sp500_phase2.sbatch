#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# InVAR-RL-UR Phase 2: SP500 25-cell sweep with the learned-gate variant.
# Single sbatch slot handles ONE fold x ALL 5 seeds via an internal loop.
# Submit 5 of these in parallel (one per fold) to fill the 25-cell matrix:
#
#   for f in 1 2 3 4 5; do
#     sbatch --export=ALL,FOLD=$f \
#       invar_rl/scripts/wulver/invar_rl_ur_sp500_phase2.sbatch
#   done
#
# Defaults reproduce the Phase 1 learned-gate mini-sweep:
#   gate_mode=learned, n_critics=7, kappa=1.0, delta_max=0.75,
#   gate_tau=0.1 (unused with learned gate but passed for completeness),
#   target_aggregator=min, 20000 timesteps, eval_freq=2000.
#
# Output layout (per cell):
#   outputs/sp500/layer2_ur/phase2_learned_gate/foldF_seedS.parquet
#   outputs/sp500/layer2_ur/phase2_learned_gate/summary/foldF_seedS.json
#
# Env vars:
#   FOLD       required, one of {1,2,3,4,5}
#   SEEDS      space-separated seed list (default "42 43 44 45 46")
#   TIMESTEPS  default 20000
#   EVAL_FREQ  default 2000
#   OUT_ROOT   default outputs/sp500/layer2_ur/phase2_learned_gate
#   LOG_ROOT   default invar_rl/logs/sp500_layer2_ur_phase2_learned_gate

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

FOLD="${FOLD:?FOLD env var required (1..5)}"
SEEDS="${SEEDS:-42 43 44 45 46}"
TIMESTEPS="${TIMESTEPS:-20000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"
OUT_ROOT="${OUT_ROOT:-outputs/sp500/layer2_ur/phase2_learned_gate}"
LOG_ROOT="${LOG_ROOT:-invar_rl/logs/sp500_layer2_ur_phase2_learned_gate}"
CKPT_ROOT="invar_rl/results/stage1/_ckpt"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$OUT_ROOT/summary"

echo "host=$(hostname) start=$(date -Is) fold=$FOLD seeds='$SEEDS' timesteps=$TIMESTEPS out_root=$OUT_ROOT"
nvidia-smi | head -8 || true

FAILED=0
for SEED in $SEEDS; do
  CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
  if [ ! -f "$CKPT" ]; then
    echo "[$(date +%H:%M:%S)] MISSING layer1 ckpt: $CKPT; skipping fold=$FOLD seed=$SEED"
    FAILED=$((FAILED + 1))
    continue
  fi

  LF="${LOG_ROOT}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] run UR fold=$FOLD seed=$SEED -> $LF"

  PYTHONPATH=$PWD python3 -u -m invar_rl.training.sp500_layer2_ur \
      --fold "$FOLD" --seed "$SEED" \
      --total-timesteps "$TIMESTEPS" \
      --eval-freq "$EVAL_FREQ" \
      --output-dir-root "$OUT_ROOT" \
      --layer1-ckpt-root "$CKPT_ROOT" \
      --layer3 invar_rl/configs/layer3.yaml \
      --stage3 invar_rl/configs/stage3.yaml \
      --n-critics 7 --kappa 1.0 --delta-max 0.75 \
      --gate-tau 0.1 \
      --target-aggregator min \
      --gate-mode learned \
      --learned-gate-hidden 32 \
      > "$LF" 2>&1 || {
        echo "[$(date +%H:%M:%S)] FAILED UR fold=$FOLD seed=$SEED; see $LF"
        FAILED=$((FAILED + 1))
        continue
      }
  echo "[$(date +%H:%M:%S)] DONE UR fold=$FOLD seed=$SEED"
done

echo "ALL CELLS COMPLETE fold=$FOLD failed=$FAILED end=$(date -Is)"
ls -la "$OUT_ROOT/summary/" 2>/dev/null | grep "fold${FOLD}_" || true
exit $FAILED
