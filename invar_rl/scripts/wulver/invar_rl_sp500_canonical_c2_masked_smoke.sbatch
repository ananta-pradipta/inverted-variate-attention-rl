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

# C2 (2026-05-27): Stage-1 MASKED FEATURE MODELING pretrain SMOKE on
# SP500 fold=2 seed=42. Replaces the canonical InfoNCE objective with
# a BERT/MAE-style masked-feature reconstruction pretext: per stock
# per day a Bernoulli(0.15) mask over the F feature channels is
# applied to the LAST-step feature row of the lookback; the encoder
# consumes the masked window and a small linear decoder (d_model -> F)
# reconstructs the original feature values. Loss = MSE on masked
# positions. Decoder discarded after Stage 1; only the encoder is
# carried into Stage 2.
#
# Fold 2 deliberately chosen: F2 rate-stress is the most sensitive
# cell to a load-bearing pretrain change (same rationale as
# B1 / B2 / C3 smokes).
#
# Smoke success criterion: L1 test rank IC in [-0.05, +0.05] sanity
# range AND the MFM Stage-1 loss decreases monotonically over the 10
# pretrain epochs. A large IC swing means the decoder is over- or
# under-fitting; revisit before the 25-cell sweep.
#
# Diff vs invar_rl_sp500_canonical_c3_sector_smoke.sbatch:
#   - Output dir outputs/sp500/layer1_c2_masked_smoke/
#   - Stage 1 PRETRAIN call swaps the C3 sector flags for:
#       --pretrain-method masked_feature
#       --pretrain-mask-ratio 0.15
#   - Stage 2 finetune is unchanged (canonical cs_mse path).
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sp500_canonical_c2_masked_smoke.sbatch

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
MASK_RATIO="${MASK_RATIO:-0.15}"

OUT_DIR="outputs/sp500/layer1_c2_masked_smoke"
LOG_DIR="invar_rl/logs/sp500_layer1_c2_masked_smoke"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD} seed=${SEED} method=masked_feature ratio=${MASK_RATIO}"
nvidia-smi | head -8 || true

# Stage 1.a: per-fold encoder pretrain with C2 MASKED FEATURE pretext ON.
ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
if [ -f "$ENC_CKPT" ]; then
  echo "[$(date +%H:%M:%S)] fold=$FOLD: encoder ckpt exists, skip pretrain"
else
  PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD (C2 MASKED FEATURE ON, ratio=${MASK_RATIO})"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed 42 \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --pretrain_only \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --output_dir "$OUT_DIR" \
    --pretrain-method masked_feature \
    --pretrain-mask-ratio "$MASK_RATIO" \
    > "$PLF" 2>&1
fi

# Stage 1.b: finetune fold=2 seed=42 with the canonical cs_mse loss.
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
    print(f'[C2 smoke] no result JSON at {out_path}')
else:
    d = json.loads(out_path.read_text())
    print('[C2 smoke] FOLD=2 SEED=42 lattice_native two_regime_val method=masked_feature')
    print(f'  test ic        = {d.get(\"ic\"):+.4f}')
    print(f'  test rank_ic   = {d.get(\"rank_ic\"):+.4f}')
    print(f'  test ndcg10    = {d.get(\"ndcg10\"):.4f}')
    print(f'  val  ic        = {d.get(\"val_ic\"):+.4f}')
    print(f'  val  rank_ic   = {d.get(\"val_rank_ic\"):+.4f}')
    print('[canonical ref] fold2 seed42 (canonical pretrain): test rank_ic +0.0028 5-seed pooled')
    print('[B1 ref       ] fold2 seed42 (HMM pretrain):       test rank_ic +0.043 5-seed pooled (B1 WIN)')
    print('[C3 ref       ] fold2 seed42 (sector pretrain):    test rank_ic STRONG WIN +0.101 pool 5-seed (C3 LEADER)')
    print('[C2 smoke gate] L1 test rank IC must be in [-0.05, +0.05] sanity range')
"

# Pretrain loss trajectory: the MFM MSE should be monotonically
# decreasing across the 10 epochs (different from InfoNCE which can
# fluctuate at lower batch counts).
echo '[C2 smoke] pretrain MSE trajectory:'
grep 'mfm_mse=' "${LOG_DIR}/fold${FOLD}_pretrain.log" 2>/dev/null || true

echo "DONE fold=${FOLD} seed=${SEED} $(date -Is)"
