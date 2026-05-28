#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# biotech NBI Phase 5 Layer 3 SAC controller training on the ENRICHED
# 22-feature panel, with K=25 per side wrapper (fair comparison vs
# K=25 baselines on the enriched panel). Mirrors
# invar_rl_biotech_nbi_enriched_layer3.sbatch but overrides the
# wrapper to equal_topk K=25.
#
# Slot layout (mod 5 = seed index, div 5 = protocol index):
#   SLOT 0..4 -> protocol=ls, seeds=42..46
#   SLOT 5..9 -> protocol=lo, seeds=42..46
#
# 10 slots in parallel, ~30 min wall per cell.

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
PROTOS=(ls lo)
SEED="${SEEDS[$((SLOT % 5))]}"
PROTO="${PROTOS[$((SLOT / 5))]}"
TIMESTEPS="${TIMESTEPS:-20000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"
EQUAL_TOPK_K="${EQUAL_TOPK_K:-25}"

OUT_ROOT="${OUT_ROOT:-outputs/biotech_nbi_enriched/layer3_k25}"
LOG_DIR="${LOG_DIR:-invar_rl/logs/$(basename "$OUT_ROOT")}"
CKPT_ROOT="${CKPT_ROOT:-outputs/biotech_nbi_enriched/layer1/_ckpt}"
mkdir -p "${OUT_ROOT}/${PROTO}/summary" "$LOG_DIR"

echo "host=$(hostname) slot=$SLOT seed=$SEED proto=$PROTO k=${EQUAL_TOPK_K} start=$(date -Is)"
nvidia-smi | head -8 || true

for F in 1 2 3 4 5; do
  CKPT="${CKPT_ROOT}/fold${F}_seed${SEED}_full.pt"
  OUT="${OUT_ROOT}/${PROTO}/fold${F}_seed${SEED}.parquet"
  SUMMARY="${OUT_ROOT}/${PROTO}/summary/fold${F}_seed${SEED}.json"
  if [ ! -f "$CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$F seed=$SEED proto=$PROTO: ckpt missing $CKPT"
    continue
  fi
  if [ -f "$OUT" ] && [ -f "$SUMMARY" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$F seed=$SEED proto=$PROTO: outputs exist"
    continue
  fi
  LF="${LOG_DIR}/fold${F}_seed${SEED}_${PROTO}.log"
  echo "[$(date +%H:%M:%S)] sac-train fold=$F seed=$SEED proto=$PROTO k=${EQUAL_TOPK_K}"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.biotech_nbi_enriched_layer3_sac \
    --fold "$F" --seed "$SEED" --protocol "$PROTO" \
    --total-timesteps "$TIMESTEPS" \
    --eval-freq "$EVAL_FREQ" \
    --output-dir-root "$OUT_ROOT" \
    --layer1-ckpt-root "$CKPT_ROOT" \
    --layer3 invar_rl/configs/layer3.yaml \
    --stage3 invar_rl/configs/stage3.yaml \
    --weighting-mode equal_topk \
    --equal-topk-k "$EQUAL_TOPK_K" \
    > "$LF" 2>&1 || {
      echo "FAILED fold=$F seed=$SEED proto=$PROTO ; see $LF"
      continue
    }
done

echo "DONE slot=$SLOT seed=$SEED proto=$PROTO $(date -Is)"
