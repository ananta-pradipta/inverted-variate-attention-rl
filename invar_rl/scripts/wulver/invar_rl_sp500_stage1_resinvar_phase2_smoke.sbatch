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

# ResInVAR-RL Phase 2 single-cell smoke (2026-05-26).
#
# Layer 1 only: regime-contrastive Stage 1 pretrain (unchanged) +
# CAE-Head target-residualized Stage 2 finetune (Phase 2 change).
# Single cell: SP500 fold=2 seed=42. Wall budget 4h covers both stages
# on a100_20g with the per-cell wall-clocks the canonical pipeline
# normally hits (Stage 1 approx 15-30 min, Stage 2 approx 20-40 min).
#
# Phase 2 stop gate (from spec section 4 Phase 2):
#   - Single-cell training completes without exceptions.
#   - L_rec final epoch < L_rec epoch 1 (allowing 10 percent noise).
#   - y_tilde-vs-y Pearson correlation < 0.7 by epoch 5.
#   - Layer 1 rank IC on the test segment in [-0.05, +0.05].
#
# Outputs:
#   outputs/sp500/layer1_resinvar_canonical_phase2/_ckpt/fold2_seed42_full.pt
#   outputs/sp500/layer1_resinvar_canonical_phase2/fold2_seed42.json
#   outputs/resinvar_rl/lattice_native/resinvar_canonical/seed42/fold2/
#     metrics.jsonl
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sp500_stage1_resinvar_phase2_smoke.sbatch

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Persist the full layer-1 state_dict so downstream Phase 3 can load it.
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

OUT_DIR="outputs/sp500/layer1_resinvar_canonical_phase2"
LOG_DIR="invar_rl/logs/sp500_layer1_resinvar_canonical_phase2"
CKPT_DIR="${OUT_DIR}/_ckpt"
METRICS_DIR="outputs/resinvar_rl/lattice_native/resinvar_canonical/seed${SEED}/fold${FOLD}"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR" "$METRICS_DIR"

echo "[INFO] host=$(hostname) start=$(date -Is) fold=${FOLD} seed=${SEED} variant=resinvar_canonical"
nvidia-smi | head -8 || true

# Stage 1.a: per-fold encoder pretrain. Phase 2 does NOT change Stage 1;
# the encoder ckpt is the same SimCLR-style InfoNCE regime-contrastive
# pretrain as canonical InVAR.
ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
if [ -f "$ENC_CKPT" ]; then
  echo "[INFO] $(date +%H:%M:%S) fold=$FOLD: encoder ckpt exists, skip pretrain"
else
  PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[INFO] $(date +%H:%M:%S) pretrain fold=$FOLD"
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

# Stage 1.b: finetune fold=2 seed=42 with CAE-Head target-residualization
# enabled. Stage 2 hyperparameters per spec section 3.1.
OUT_JSON="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
FULL_CKPT="${CKPT_DIR}/fold${FOLD}_seed${SEED}_full.pt"
if [ -f "$OUT_JSON" ] && [ -f "$FULL_CKPT" ]; then
  echo "[INFO] $(date +%H:%M:%S) skip finetune: $OUT_JSON + $FULL_CKPT present"
else
  LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[INFO] $(date +%H:%M:%S) finetune fold=$FOLD seed=$SEED cae_head=ON"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --skip_pretrain \
    --finetune_epochs "$FINETUNE_EPOCHS" \
    --output_dir "$OUT_DIR" \
    --cae-head-enabled \
    --cae-head-k-latent 3 \
    --cae-head-lambda-rec 0.1 \
    --cae-head-lambda-orth 0.01 \
    --cae-head-lr 0.001 \
    --cae-head-metrics-dir "$METRICS_DIR" \
    > "$LF" 2>&1
fi

echo "[INFO] DONE fold=${FOLD} seed=${SEED} $(date -Is)"
