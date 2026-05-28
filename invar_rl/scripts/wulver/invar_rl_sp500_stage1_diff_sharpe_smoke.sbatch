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

# Option C (differentiable Sharpe surrogate at L1 finetune) smoke
# (2026-05-26). Single-cell SP500 fold=2 seed=42 sanity check that
# the new loss_config=diff_sharpe path trains end-to-end and produces
# a Layer-1 rank IC comparable to the canonical cs_mse baseline.
#
# Fold 2 deliberately chosen: it is the F2 rate-stress fold where the
# canonical InVAR scores rank IC +0.0028 and the L/S Sharpe is the
# weakest (-0.229 pool baseline component). Smoke success criterion:
# F2 not catastrophically worse (within +/- 0.05 of canonical IC).
#
# Diff vs invar_rl_sp500_stage1_retrain.sbatch:
#   - Only FOLD=2 SEED=42 (one cell)
#   - Time budget 4h
#   - Output dir outputs/sp500/layer1_diff_sharpe_smoke/
#   - --loss-config diff_sharpe --diff-sharpe-weight 0.2
#     --diff-sharpe-k 50 --diff-sharpe-temperature 0.1
#     --diff-sharpe-batch-days 16 on the finetune call
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sp500_stage1_diff_sharpe_smoke.sbatch

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

OUT_DIR="outputs/sp500/layer1_diff_sharpe_smoke"
LOG_DIR="invar_rl/logs/sp500_layer1_diff_sharpe_smoke"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD} seed=${SEED} loss=diff_sharpe"
nvidia-smi | head -8 || true

# Stage 1.a: per-fold encoder pretrain. The Stage-1 SSL objective is
# SimCLR InfoNCE and is loss_config agnostic; pretrain depends only on
# the fold, not seed or loss_config.
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

# Stage 1.b: finetune fold=2 seed=42 with --loss-config diff_sharpe.
OUT_JSON="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
FULL_CKPT="${CKPT_DIR}/fold${FOLD}_seed${SEED}_full.pt"
if [ -f "$OUT_JSON" ] && [ -f "$FULL_CKPT" ]; then
  echo "[$(date +%H:%M:%S)] skip finetune: $OUT_JSON + $FULL_CKPT present"
else
  LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] finetune fold=$FOLD seed=$SEED loss=diff_sharpe"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --skip_pretrain \
    --finetune_epochs "$FINETUNE_EPOCHS" \
    --output_dir "$OUT_DIR" \
    --loss-config diff_sharpe \
    --diff-sharpe-weight 0.2 \
    --diff-sharpe-k 50 \
    --diff-sharpe-temperature 0.1 \
    --diff-sharpe-batch-days 16 \
    > "$LF" 2>&1
fi

# Result rollup so the .out file shows the headline numbers.
python3 -c "
import json
from pathlib import Path
out_path = Path('${OUT_DIR}') / 'fold${FOLD}_seed${SEED}.json'
if not out_path.exists():
    print(f'[diff_sharpe smoke] no result JSON at {out_path}')
else:
    d = json.loads(out_path.read_text())
    print('[diff_sharpe smoke] FOLD=2 SEED=42 lattice_native two_regime_val')
    print(f'  test ic        = {d.get(\"ic\"):+.4f}')
    print(f'  test rank_ic   = {d.get(\"rank_ic\"):+.4f}')
    print(f'  test ndcg10    = {d.get(\"ndcg10\"):.4f}')
    print(f'  val  ic        = {d.get(\"val_ic\"):+.4f}')
    print(f'  val  rank_ic   = {d.get(\"val_rank_ic\"):+.4f}')
    print('[canonical ref] fold2 seed42 (cs_mse): test rank_ic +0.0028 5seed pooled')
"

echo "DONE fold=${FOLD} seed=${SEED} $(date -Is)"
