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

# A1 (2026-05-27): Stage-1 SEQUENTIAL pretrain regime -> sector
# SMOKE on SP500 fold=2 seed=42. Stage 1a uses the canonical k-means-8
# day-level InfoNCE selector for 10 epochs; Stage 1b continues training
# the SAME encoder weights under the C3 per-stock same-sector InfoNCE
# selector for another 10 epochs. Stage 2 finetune is unchanged
# (canonical cs_mse).
#
# Hypothesis: combine B1-style regime coverage (F2 robustness) with
# C3-style sector clustering (F1/F4 lift). Multi-stage curriculum.
#
# Fold 2 deliberately chosen: F2 rate-stress is the most sensitive
# cell to a load-bearing pretrain change (same rationale as B1 / C3
# smokes).
#
# Smoke success criterion: L1 test rank IC in [-0.05, +0.05] sanity
# range AND both stages execute (regime + sector) with finite loss
# across all reported epochs.
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sp500_canonical_a1_seq_regime_sector_smoke.sbatch

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

FOLD=2
SEED=42
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"

OUT_DIR="outputs/sp500/layer1_a1_seq_regime_sector_smoke"
LOG_DIR="invar_rl/logs/sp500_layer1_a1_seq_regime_sector_smoke"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD} seed=${SEED} stages=regime,sector"
nvidia-smi | head -8 || true

# Stage 1: per-fold encoder pretrain with A1 SEQUENTIAL stages.
ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
if [ -f "$ENC_CKPT" ]; then
  echo "[$(date +%H:%M:%S)] fold=$FOLD: encoder ckpt exists, skip pretrain"
else
  PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD (A1 SEQUENTIAL regime,sector)"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed 42 \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --pretrain_only \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --output_dir "$OUT_DIR" \
    --pretrain-stages regime,sector \
    --pretrain-sector-universe-id sp500 \
    > "$PLF" 2>&1
  # Quick verification both stages ran.
  N_STAGES=$(grep -c "\[InVAR-clpretrain S1\] A1 stage" "$PLF" || true)
  echo "[A1 smoke] pretrain log stages reported: ${N_STAGES} (expect 2)"
fi

# Stage 2: finetune fold=2 seed=42 with the canonical cs_mse loss.
OUT_JSON="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
FULL_CKPT="${CKPT_DIR}/fold${FOLD}_seed${SEED}_full.pt"
if [ -f "$OUT_JSON" ] && [ -f "$FULL_CKPT" ]; then
  echo "[$(date +%H:%M:%S)] skip finetune: $OUT_JSON + $FULL_CKPT present"
else
  LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] finetune fold=$FOLD seed=$SEED (canonical cs_mse)"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --skip_pretrain \
    --finetune_epochs "$FINETUNE_EPOCHS" \
    --output_dir "$OUT_DIR" \
    > "$LF" 2>&1
fi

# Result rollup so the .out file shows the headline numbers.
python3 -c "
import json
from pathlib import Path
out_path = Path('${OUT_DIR}') / 'fold${FOLD}_seed${SEED}.json'
if not out_path.exists():
    print(f'[A1 smoke] no result JSON at {out_path}')
else:
    d = json.loads(out_path.read_text())
    print('[A1 smoke] FOLD=2 SEED=42 lattice_native two_regime_val stages=regime,sector')
    print(f'  test ic        = {d.get(\"ic\"):+.4f}')
    print(f'  test rank_ic   = {d.get(\"rank_ic\"):+.4f}')
    print(f'  test ndcg10    = {d.get(\"ndcg10\"):.4f}')
    print(f'  val  ic        = {d.get(\"val_ic\"):+.4f}')
    print(f'  val  rank_ic   = {d.get(\"val_rank_ic\"):+.4f}')
    print('[canonical ref] fold2 seed42 (canonical pretrain): test rank_ic +0.0028 5-seed pooled')
    print('[C3 ref       ] fold2 seed42 (C3 sector pretrain): pool +1.046 (WIN)')
    print('[B1 ref       ] fold2 seed42 (B1 HMM pretrain): pool +0.979 (WIN)')
    print('[A1 smoke gate] L1 test rank IC must be in [-0.05, +0.05] sanity range')
    print('[A1 smoke gate] both stages must execute (look for [A1 stage 1/2] and [A1 stage 2/2])')
"

echo "DONE fold=${FOLD} seed=${SEED} $(date -Is)"
