#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# F2 fundamental L1 listwise rank loss upgrade smoke (2026-05-26).
#
# Single-cell SP500 fold=1 seed=42 sanity check that the new
# loss_config=listmle path trains end-to-end and produces a Layer-1
# rank IC comparable to the canonical cs_mse baseline. Compare to
# reports/sp500/stage1_retrain_2026-05-23.md (F1 seed42 canonical
# rank IC +0.0426). DO NOT branch to a 25-cell sweep without a
# positive smoke verdict.
#
# Diff vs invar_rl_sp500_stage1_retrain.sbatch:
#   - Only FOLD=1, SEED=42 (one cell)
#   - Time budget 4h (one cell instead of 5 seeds)
#   - Output dir outputs/sp500/layer1_f2_listmle_smoke/
#   - --loss-config listmle on the finetune call
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sp500_stage1_f2_listmle_smoke.sbatch

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Persist the full layer-1 state_dict so the wrapper / SAC eval can load it.
export INVAR_SAVE_FULL_STATE=1

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

FOLD=1
SEED=42
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"

OUT_DIR="outputs/sp500/layer1_f2_listmle_smoke"
LOG_DIR="invar_rl/logs/sp500_layer1_f2_listmle_smoke"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD} seed=${SEED} loss=listmle"
nvidia-smi | head -8 || true

# Stage 1.a: per-fold encoder pretrain (cs_mse / listmle agnostic; the
# Stage-1 SSL objective is the SimCLR-style InfoNCE, not the supervised
# ranking loss). Pretrain only depends on the fold, not the seed or
# loss_config.
ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
if [ -f "$ENC_CKPT" ]; then
  echo "[$(date +%H:%M:%S)] fold=$FOLD: encoder ckpt exists, skip pretrain"
else
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

# Stage 1.b: finetune fold=1 seed=42 with --loss-config listmle.
OUT_JSON="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
FULL_CKPT="${CKPT_DIR}/fold${FOLD}_seed${SEED}_full.pt"
if [ -f "$OUT_JSON" ] && [ -f "$FULL_CKPT" ]; then
  echo "[$(date +%H:%M:%S)] skip finetune: $OUT_JSON + $FULL_CKPT present"
else
  LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] finetune fold=$FOLD seed=$SEED loss=listmle"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --skip_pretrain \
    --finetune_epochs "$FINETUNE_EPOCHS" \
    --output_dir "$OUT_DIR" \
    --loss-config listmle \
    > "$LF" 2>&1
fi

echo "DONE fold=${FOLD} seed=${SEED} $(date -Is)"
