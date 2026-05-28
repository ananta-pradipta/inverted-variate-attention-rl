#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Option F (SAC curriculum) per-cell finetune.
# For each (fold, seed) in {1..5} x {42..46}, warm-starts SAC from the
# corresponding seed's calm-pretrain ckpt at
# outputs/sac_curriculum/sp500/calm_pretrain/seed{S}/sac_ckpt.zip and
# finetunes for 20k more timesteps on the cell's own train tape
# (equal_l2 wrapper, K=50). Eval on test tape; daily tape persisted.
#
# 5 parallel slots, round-robin over the 25 cells.
#
# Submit (5 parallel slots):
#   for SLOT in 0 1 2 3 4; do
#     sbatch --export=ALL,SLOT=$SLOT \
#       invar_rl/scripts/wulver/sac_curriculum_per_cell_finetune.sbatch
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

SLOT="${SLOT:?SLOT must be set via --export=SLOT=0..4}"

OUT_ROOT="outputs/sac_curriculum/sp500/per_cell_finetune"
TAPE_ROOT="${OUT_ROOT}/daily_tape"
LOG_ROOT="invar_rl/logs/sac_curriculum_per_cell_finetune"
CKPT_ROOT="invar_rl/results/stage1/_ckpt"
PRETRAIN_ROOT="outputs/sac_curriculum/sp500/calm_pretrain"
mkdir -p "$OUT_ROOT" "$TAPE_ROOT" "$LOG_ROOT"

TUPLES=()
for FOLD in 1 2 3 4 5; do
  for SEED in 42 43 44 45 46; do
    TUPLES+=("${FOLD}|${SEED}")
  done
done
N_TUPLES=${#TUPLES[@]}
N_SLOTS=5

echo "host=$(hostname) start=$(date -Is) slot=$SLOT total_tuples=$N_TUPLES"
nvidia-smi | head -8 || true

for i in $(seq 0 $((N_TUPLES - 1))); do
  if [ $((i % N_SLOTS)) -ne "$SLOT" ]; then
    continue
  fi
  IFS='|' read -r FOLD SEED <<< "${TUPLES[$i]}"
  L1_CKPT="${CKPT_ROOT}/fold${FOLD}_seed${SEED}_full.pt"
  WARM_CKPT="${PRETRAIN_ROOT}/seed${SEED}/sac_ckpt.zip"
  OUT_JSON="${OUT_ROOT}/fold${FOLD}_seed${SEED}.json"
  LF="${LOG_ROOT}/fold${FOLD}_seed${SEED}.log"

  if [ ! -f "$L1_CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED: L1 ckpt missing $L1_CKPT"
    continue
  fi
  if [ ! -f "$WARM_CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED: warm-start ckpt missing $WARM_CKPT"
    continue
  fi
  if [ -f "$OUT_JSON" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED (json present)"
    continue
  fi
  echo "[$(date +%H:%M:%S)] finetune fold=$FOLD seed=$SEED warm=$WARM_CKPT"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage3_rl_ablation \
    --ablation equal_l2 \
    --fold "$FOLD" --seed "$SEED" \
    --layer1-ckpt "$L1_CKPT" \
    --output-dir "$OUT_ROOT" \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --equal-topk-k 50 \
    --methods sac \
    --save-daily-tape \
    --daily-tape-dir "$TAPE_ROOT" \
    --warm-start-ckpt "$WARM_CKPT" \
    --warm-start-timesteps 20000 \
    > "$LF" 2>&1 || {
      echo "FAILED fold=$FOLD seed=$SEED; see $LF"
      tail -50 "$LF"
      continue
    }
done

echo "DONE slot=${SLOT} $(date -Is)"
ls -la "$OUT_ROOT/" 2>/dev/null || true
