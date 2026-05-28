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

# B1 (2026-05-27): Stage-1 HMM regime posterior contrastive selector
# SMOKE on SP500 fold=2 seed=42. Replaces the canonical L2-nearest-
# neighbour selector over the 14-d standardised episode-key fingerprint
# with a Gaussian HMM (hmmlearn preferred, sklearn GMM fallback) posterior
# cosine-similarity selector; positives = in-batch days whose posterior
# cosine similarity to the anchor is >= --pretrain-hmm-positive-threshold.
#
# Fold 2 deliberately chosen: F2 rate-stress is where the canonical
# InVAR scores rank IC +0.0028 and the L/S Sharpe is weakest, so it is
# the most sensitive cell to a load-bearing pretrain change. (Same
# rationale as the B2 smoke.)
#
# Smoke success criterion: L1 test rank IC in [-0.05, +0.05] sanity
# range AND HMM posteriors are well-formed (sum to 1, no NaN). A large
# IC swing means the HMM selector is over- or under-clustering; tune
# n_states or threshold before the 25-cell sweep.
#
# Diff vs invar_rl_sp500_canonical_b2_pretrain_aux_smoke.sbatch:
#   - Output dir outputs/sp500/layer1_b1_hmm_smoke/
#   - Stage 1 PRETRAIN call swaps the aux-head flags for:
#       --pretrain-regime-method hmm
#       --pretrain-hmm-n-states 4
#       --pretrain-hmm-positive-threshold 0.7
#       --pretrain-hmm-universe-id sp500
#   - Stage 2 finetune is unchanged (canonical cs_mse path).
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sp500_canonical_b1_hmm_smoke.sbatch

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
HMM_N_STATES="${HMM_N_STATES:-4}"
HMM_THRESH="${HMM_THRESH:-0.7}"

OUT_DIR="outputs/sp500/layer1_b1_hmm_smoke"
LOG_DIR="invar_rl/logs/sp500_layer1_b1_hmm_smoke"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD} seed=${SEED} n_states=${HMM_N_STATES} thresh=${HMM_THRESH}"
nvidia-smi | head -8 || true

# Stage 1.a: per-fold encoder pretrain with B1 HMM regime selector ON.
ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
if [ -f "$ENC_CKPT" ]; then
  echo "[$(date +%H:%M:%S)] fold=$FOLD: encoder ckpt exists, skip pretrain"
else
  PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD (B1 HMM ON, n_states=$HMM_N_STATES, thresh=$HMM_THRESH)"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed 42 \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --pretrain_only \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --output_dir "$OUT_DIR" \
    --pretrain-regime-method hmm \
    --pretrain-hmm-n-states "$HMM_N_STATES" \
    --pretrain-hmm-positive-threshold "$HMM_THRESH" \
    --pretrain-hmm-universe-id sp500 \
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
    print(f'[B1 smoke] no result JSON at {out_path}')
else:
    d = json.loads(out_path.read_text())
    print('[B1 smoke] FOLD=2 SEED=42 lattice_native two_regime_val n_states=${HMM_N_STATES} thresh=${HMM_THRESH}')
    print(f'  test ic        = {d.get(\"ic\"):+.4f}')
    print(f'  test rank_ic   = {d.get(\"rank_ic\"):+.4f}')
    print(f'  test ndcg10    = {d.get(\"ndcg10\"):.4f}')
    print(f'  val  ic        = {d.get(\"val_ic\"):+.4f}')
    print(f'  val  rank_ic   = {d.get(\"val_rank_ic\"):+.4f}')
    print('[canonical ref] fold2 seed42 (canonical pretrain): test rank_ic +0.0028 5-seed pooled')
    print('[B1 smoke gate] L1 test rank IC must be in [-0.05, +0.05] sanity range')

# Also dump the posterior cache summary so the audit can verify
# train-only fit + shape.
post_path = Path('cache/pretrain_improvements/hmm_regime/sp500/fold${FOLD}/posteriors.parquet')
if post_path.exists():
    import pandas as pd
    df = pd.read_parquet(post_path)
    print(f'[B1 smoke] posterior cache: {post_path}')
    print(f'  rows={len(df)} backend={df[\"backend\"].iloc[0]} n_states={int(df[\"n_states\"].iloc[0])} n_train_days={int(df[\"n_train_days\"].iloc[0])}')
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    row_sums = df[prob_cols].sum(axis=1)
    print(f'  row_sum range = [{row_sums.min():.6f}, {row_sums.max():.6f}] (must be ~1.0)')
else:
    print(f'[B1 smoke] WARN: posterior cache NOT FOUND at {post_path}')
"

echo "DONE fold=${FOLD} seed=${SEED} $(date -Is)"
