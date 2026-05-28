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

# Robust-InVAR-RL Phase 1 smoke: SP500 fold=2 seed=42 only.
# Stage 1 (Layer 1) retrain with group-DRO + top-bottom loss.
# Single (fold, seed) cell on the canonical lattice_native panel; matches
# the Phase 0 anchor cell (fold=2 has the F2 rate-stress test segment).
#
# Verifies:
# - Stage 1 finetune completes without exceptions.
# - q_state shifts toward worst-loss group (printed every epoch).
# - Test rank IC within +/- 0.005 of canonical F2 reference.
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sp500_stage1_robust_phase1_smoke.sbatch

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Persist full layer-1 state_dict so downstream L2L3 can load it.
export INVAR_SAVE_FULL_STATE=1

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

FOLD=2
SEED=42
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
ETA_GDRO="${ETA_GDRO:-0.05}"
LAMBDA_TB="${LAMBDA_TB:-0.10}"
M_TB="${M_TB:-50}"

OUT_DIR="invar_rl/results/stage1_robust_phase1_smoke"
LOG_DIR="invar_rl/logs/stage1_robust_phase1_smoke"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD} seed=${SEED}"
echo "gdro: eta=${ETA_GDRO} lambda_tb=${LAMBDA_TB} M=${M_TB}"
nvidia-smi | head -8 || true

# Stage 1.a: per-fold encoder pretrain (regime-contrastive; NOT changed
# by group-DRO, which only affects Stage 2 finetune). Skip if encoder
# ckpt already present.
ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
if [ ! -f "$ENC_CKPT" ]; then
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
else
  echo "[$(date +%H:%M:%S)] encoder ckpt present; skip pretrain"
fi

# Stage 1.b: finetune with group-DRO + top-bottom loss.
LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
echo "[$(date +%H:%M:%S)] gdro finetune fold=$FOLD seed=$SEED"
PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
  --fold "$FOLD" --seed "$SEED" \
  --panel_kind lattice_native \
  --panel_end 2025-12-31 \
  --two_regime_val \
  --skip_pretrain \
  --finetune_epochs "$FINETUNE_EPOCHS" \
  --output_dir "$OUT_DIR" \
  --use-group-dro \
  --eta-gdro "$ETA_GDRO" \
  --lambda-top-bottom "$LAMBDA_TB" \
  --m-top-bottom "$M_TB" \
  --gdro-universe-id sp500 \
  > "$LF" 2>&1

echo "DONE smoke $(date -Is)"
ls -la "$OUT_DIR/" "$CKPT_DIR/" 2>/dev/null || true
echo "--- tail of finetune log ---"
tail -40 "$LF" || true
