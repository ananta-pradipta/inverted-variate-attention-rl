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

# A2 (2026-05-27): Sequential pretrain regime -> co-movement clustering
# SMOKE on SP500 fold=2 seed=42. Stage 1a = canonical regime-coherent
# InfoNCE (10 epochs); Stage 1b = co-movement-clustered per-stock SupCon
# InfoNCE (5 epochs by default) where positives = same-day same-cluster
# peers and the cluster ids come from a per-fold 252-day rolling
# correlation k-means (K=8) cached at
# cache/pretrain_improvements/comovement/sp500/foldF/cluster_ids.parquet.
#
# Fold 2 deliberately chosen: F2 rate-stress is the most sensitive cell
# to a load-bearing pretrain change (same rationale as B1 / B2 / C3
# smokes).
#
# Smoke success criterion: L1 test rank IC in [-0.05, +0.05] sanity
# range AND both pretrain stages complete AND the co-movement cluster
# parquet contains 8 clusters with every cluster size >= 5. A large IC
# swing means the curriculum is over- or under-clustering; revisit
# before the 25-cell sweep.
#
# Diff vs invar_rl_sp500_canonical_c3_sector_smoke.sbatch:
#   - Output dir outputs/sp500/layer1_a2_seq_regime_comove_smoke/
#   - Stage 1 PRETRAIN call swaps the C3 sector flags for:
#       --pretrain-stages regime,comovement
#       --pretrain-comovement-universe-id sp500
#       --pretrain-comovement-n-clusters 8
#       --pretrain-comovement-window 252
#       --pretrain-comovement-epochs 5
#   - Stage 2 finetune is unchanged (canonical cs_mse path).
#
# Submit:
#   sbatch invar_rl/scripts/wulver/invar_rl_sp500_canonical_a2_seq_regime_comove_smoke.sbatch

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
COMOVE_EPOCHS="${COMOVE_EPOCHS:-5}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
COMOVE_K="${COMOVE_K:-8}"
COMOVE_WINDOW="${COMOVE_WINDOW:-252}"

OUT_DIR="outputs/sp500/layer1_a2_seq_regime_comove_smoke"
LOG_DIR="invar_rl/logs/sp500_layer1_a2_seq_regime_comove_smoke"
CKPT_DIR="${OUT_DIR}/_ckpt"
mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD} seed=${SEED} stages=regime,comovement K=${COMOVE_K}"
nvidia-smi | head -8 || true

# Stage 1.a + 1.b: per-fold encoder pretrain with the A2 sequential
# regime -> comovement curriculum. Stage 1a uses the canonical k-means-8
# regime selector (10 epochs); Stage 1b loads the Stage-1a encoder and
# continues training under the per-day per-stock SupCon InfoNCE with
# co-movement cluster positives (COMOVE_EPOCHS epochs).
ENC_CKPT="${CKPT_DIR}/fold${FOLD}_encoder.pt"
if [ -f "$ENC_CKPT" ]; then
  echo "[$(date +%H:%M:%S)] fold=$FOLD: encoder ckpt exists, skip pretrain"
else
  PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] pretrain fold=$FOLD (A2 stages=regime,comovement, K=$COMOVE_K)"
  PYTHONPATH=$PWD python3 -u -m invar_rl.training.stage1_rank \
    --fold "$FOLD" --seed 42 \
    --panel_kind lattice_native \
    --panel_end 2025-12-31 \
    --two_regime_val \
    --pretrain_only \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --output_dir "$OUT_DIR" \
    --pretrain-stages regime,comovement \
    --pretrain-comovement-universe-id sp500 \
    --pretrain-comovement-n-clusters "$COMOVE_K" \
    --pretrain-comovement-window "$COMOVE_WINDOW" \
    --pretrain-comovement-epochs "$COMOVE_EPOCHS" \
    > "$PLF" 2>&1
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

# Result rollup + cluster cache health check.
python3 -c "
import json
from pathlib import Path
out_path = Path('${OUT_DIR}') / 'fold${FOLD}_seed${SEED}.json'
if not out_path.exists():
    print(f'[A2 smoke] no result JSON at {out_path}')
else:
    d = json.loads(out_path.read_text())
    print('[A2 smoke] FOLD=2 SEED=42 lattice_native two_regime_val stages=regime,comovement')
    print(f'  test ic        = {d.get(\"ic\"):+.4f}')
    print(f'  test rank_ic   = {d.get(\"rank_ic\"):+.4f}')
    print(f'  test ndcg10    = {d.get(\"ndcg10\"):.4f}')
    print(f'  val  ic        = {d.get(\"val_ic\"):+.4f}')
    print(f'  val  rank_ic   = {d.get(\"val_rank_ic\"):+.4f}')
    print('[canonical ref] fold2 seed42 (canonical regime pretrain): test rank_ic ~+0.0028 5-seed pooled')
    print('[C3 ref       ] fold2 seed42 (C3 sector pretrain):        test rank_ic +1.046 pool (C3 WIN)')
    print('[A2 smoke gate] L1 test rank IC must be in [-0.05, +0.05] sanity range')

co_path = Path('cache/pretrain_improvements/comovement/sp500/fold${FOLD}/cluster_ids.parquet')
if co_path.exists():
    import pandas as pd
    df = pd.read_parquet(co_path)
    print(f'[A2 smoke] cluster cache: {co_path}')
    print(f'  rows={len(df)} n_clusters={df[\"cluster_id\"].nunique()}')
    sizes = df['cluster_id'].value_counts().sort_index().to_dict()
    print(f'  per_cluster_counts={sizes}')
    min_size = min(sizes.values())
    print(f'  min_cluster_size={min_size} (gate: >= 5)')
    if min_size < 5:
        print('[A2 smoke] [ERR] degenerate clusters (size < 5)')
else:
    print(f'[A2 smoke] WARN: cluster cache NOT FOUND at {co_path}')
"

echo "DONE fold=${FOLD} seed=${SEED} $(date -Is)"
