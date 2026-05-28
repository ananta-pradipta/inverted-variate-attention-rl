#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Option F (SAC curriculum) calm-fold pretrain.
# For each seed in {42..46}, builds the concatenated F1+F3+F4 train tape
# (canonical L1 forward + equal_topk K=50 wrapper) and trains SAC for
# 20k timesteps. Writes outputs/sac_curriculum/sp500/calm_pretrain/
# seed{S}/sac_ckpt.zip plus meta.json with wall-time.
#
# Single sbatch with internal seed loop (per the Wulver single-job rule).
#
# Submit:
#   sbatch invar_rl/scripts/wulver/sac_curriculum_calm_pretrain.sbatch

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

OUT_ROOT="outputs/sac_curriculum/sp500/calm_pretrain"
LOG_ROOT="invar_rl/logs/sac_curriculum_calm_pretrain"
CKPT_ROOT="invar_rl/results/stage1/_ckpt"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

for SEED in 42 43 44 45 46; do
  CKPT_OUT="${OUT_ROOT}/seed${SEED}/sac_ckpt.zip"
  LF="${LOG_ROOT}/seed${SEED}.log"

  if [ -f "$CKPT_OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip seed=$SEED (ckpt present)"
    continue
  fi

  # Sanity check: all three calm-fold L1 ckpts must exist.
  MISSING=0
  for F in 1 3 4; do
    if [ ! -f "${CKPT_ROOT}/fold${F}_seed${SEED}_full.pt" ]; then
      echo "[$(date +%H:%M:%S)] missing L1 ckpt fold${F}_seed${SEED}"
      MISSING=1
    fi
  done
  if [ "$MISSING" -eq 1 ]; then
    echo "[$(date +%H:%M:%S)] skip seed=$SEED (missing calm L1 ckpts)"
    continue
  fi

  echo "[$(date +%H:%M:%S)] calm-pretrain seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.sac_curriculum_pretrain \
    --seed "$SEED" \
    --layer1-ckpt-root "$CKPT_ROOT" \
    --output-dir "$OUT_ROOT" \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --equal-topk-k 50 \
    --total-timesteps 20000 \
    > "$LF" 2>&1 || {
      echo "FAILED seed=$SEED; see $LF"
      tail -50 "$LF"
      continue
    }
done

echo "DONE $(date -Is)"
ls -la "$OUT_ROOT/"*/ 2>/dev/null || true
