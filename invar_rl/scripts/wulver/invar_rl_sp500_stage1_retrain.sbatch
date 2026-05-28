#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Path B Phase B3: re-train SP500 Layer 1 (canonical InVAR = bankless +
# clpretrain) on the rebuilt 2755-day lattice panel. One sbatch slot
# per FOLD (5 slots, 5 seeds serial in each) per
# [[feedback_wulver_single_job]]. Generic job name per
# [[feedback_wulver_job_names]].
#
# Faithful reproduction of invar_rl_stage1.sbatch (May-20 canonical
# InVAR run, [[project_invar_canonical_2026-05-19]]): same
# panel_kind=lattice_native, two_regime_val, 10 pretrain epochs, 10
# finetune epochs, INVAR_SAVE_FULL_STATE=1. The CLI diff vs that
# sbatch is the explicit --panel_end 2025-12-31 (matches the default
# in stage1_rank.py) and a skip-if-already-extended guard so we can
# re-run this sbatch idempotently.
#
# Submit (one job per fold, 5 jobs total):
#   for F in 1 2 3 4 5; do
#     sbatch --export=ALL,FOLD=$F \
#       invar_rl/scripts/wulver/invar_rl_sp500_stage1_retrain.sbatch
#   done

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Persist the full layer-1 state_dict so Ablation 6 can load it.
export INVAR_SAVE_FULL_STATE=1

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

FOLD="${FOLD:?FOLD must be set via --export=FOLD=<1..5>}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"

OUT_DIR="invar_rl/results/stage1"
LOG_DIR="invar_rl/logs/stage1_retrain"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD}"
nvidia-smi | head -8 || true

# Returns 0 if the ckpt at $1 was already retrained on the 2755-day panel.
ckpt_already_extended() {
  local CKPT="$1"
  if [ ! -f "$CKPT" ]; then return 1; fi
  python3 -c "
import torch, sys
try:
    c = torch.load('$CKPT', map_location='cpu', weights_only=False)
    pe = str(c.get('panel_end', ''))
    nf = int(c.get('n_features', -1))
    ok = (pe == '2025-12-31' and nf == 26 and
          c['state_dict']['day_memory.mem_keys'].shape[0] == 2755)
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
"
}

# Archive existing fold ckpts before overwriting (one-shot per fold).
ARCH_DIR="${CKPT_DIR}_archive_20260523"
mkdir -p "$ARCH_DIR"
NEED_RETRAIN=0
for S in 42 43 44 45 46; do
  FULL_CKPT="${CKPT_DIR}/fold${FOLD}_seed${S}_full.pt"
  if ! ckpt_already_extended "$FULL_CKPT"; then
    NEED_RETRAIN=1
  fi
done
if [ "$NEED_RETRAIN" -eq 1 ]; then
  for f in "${CKPT_DIR}/fold${FOLD}_encoder.pt" \
           "${CKPT_DIR}/fold${FOLD}_seed42_full.pt" \
           "${CKPT_DIR}/fold${FOLD}_seed43_full.pt" \
           "${CKPT_DIR}/fold${FOLD}_seed44_full.pt" \
           "${CKPT_DIR}/fold${FOLD}_seed45_full.pt" \
           "${CKPT_DIR}/fold${FOLD}_seed46_full.pt"; do
    if [ -f "$f" ] && [ ! -f "${ARCH_DIR}/$(basename "$f")" ]; then
      cp -p "$f" "${ARCH_DIR}/"
    fi
  done
fi

# Stage 1.a: per-fold encoder pretrain. Skip if all 5 seed _full.pt
# ckpts in this fold are already retrained (encoder is implicitly OK).
ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
ALL_EXT=1
for S in 42 43 44 45 46; do
  if ! ckpt_already_extended "${CKPT_DIR}/fold${FOLD}_seed${S}_full.pt"; then
    ALL_EXT=0
    break
  fi
done
if [ "$ALL_EXT" -eq 1 ]; then
  echo "[$(date +%H:%M:%S)] fold=$FOLD: all 5 _full.pt are 2755-day; skip pretrain"
else
  # Force fresh pretrain by removing the old encoder ckpt (it points at
  # the 2003-day panel encoder weights).
  if [ -f "$ENC_CKPT" ]; then
    rm -f "$ENC_CKPT"
  fi
  PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed 42 \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --pretrain_only \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --output_dir "$OUT_DIR" \
    > "$PLF" 2>&1
fi

# Stage 1.b: finetune the 5 seeds against the (now extended) fold ckpt.
run_finetune() {
  local SEED="$1"
  local FULL_CKPT="${CKPT_DIR}/fold${FOLD}_seed${SEED}_full.pt"
  local OUT_JSON="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if ckpt_already_extended "$FULL_CKPT" && [ -f "$OUT_JSON" ]; then
    echo "[$(date +%H:%M:%S)] skip fold=$FOLD seed=$SEED (extended ckpt + json present)"
    return
  fi
  local LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] finetune fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --skip_pretrain \
    --finetune_epochs "$FINETUNE_EPOCHS" \
    --output_dir "$OUT_DIR" \
    > "$LF" 2>&1
}

for S in 42 43 44 45 46; do
  run_finetune "$S"
done

echo "DONE fold=${FOLD} $(date -Is)"
